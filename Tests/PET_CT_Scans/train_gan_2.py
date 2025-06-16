import csv
from datetime import datetime
import os
import torch
import torch.optim as optim
import torch.nn as nn
from gan_module import Generator, Discriminator
from utils.loss import generator_loss, dice_loss
import torch.nn.functional as F
from utils.data_loader import create_data_loader
from itertools import cycle
from utils.utils import *
import numpy as np
from bosonsampler import BosonLatentGenerator, BosonSamplerTorch
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import torchvision.transforms as transforms
import torchvision.models as models
import time
import multiprocessing as mp
from fvcore.nn import FlopCountAnalysis
try:
    from fvcore.nn.jit_handles import generic_elementwise_flop_count
except ImportError:
    def generic_elementwise_flop_count(inputs, outputs):
        # Treat every output element as a single multiply‑add
        return dict(flops=outputs[0].numel())

# ------------------------------------------------------------------
#  Robust custom-op registration  – works on all fvcore versions
# ------------------------------------------------------------------
def _register_op(op_name, handle=generic_elementwise_flop_count):
    """
    Register <op_name> so FlopCountAnalysis can count it.

    Older fvcore builds have no 'register_op' method.  In that case we
    simply skip registration—FlopCountAnalysis will fall back to 0 FLOPs,
    which is still better than crashing.
    """
    register = getattr(FlopCountAnalysis, "register_op", None)
    if callable(register):
        register(op_name, handle)

# Element-wise activations: 1 FLOP per tensor element
for _op in (
    "aten::leaky_relu_",  # in-place
    "aten::leaky_relu",   # out-of-place
    "aten::tanh",
    "aten::sigmoid",
):
    _register_op(_op)

# Padding op – memory only, so count 0 FLOPs
_register_op("aten::pad", lambda *args, **kwargs: dict())

# --------------------------------------------------------------------
#  Helper: FLOPs for one training step  (G-fwd + D-real + D-fake)
# --------------------------------------------------------------------
def _step_flops(generator, discriminator, z, x):
    g_fwd  = FlopCountAnalysis(generator, z).total()
    d_real = FlopCountAnalysis(discriminator, x).total()
    d_fake = FlopCountAnalysis(discriminator, generator(z)).total()
    return g_fwd + d_real + d_fake

# ------------------------------------------------------------------
#  Timing helper – average ms per latent
# ------------------------------------------------------------------
def measure_sampler_time(fn, *, n_latents=128, repeats=20, device_str="cpu"):
    """
    Return average milliseconds per latent for `fn(batch_size)`.
    """
    _ = fn(8)  # warm-up
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device_str == "mps" and hasattr(torch, "mps"):
        try: torch.mps.synchronize()
        except AttributeError: pass

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn(n_latents)
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device_str == "mps" and hasattr(torch, "mps"):
            try: torch.mps.synchronize()
            except AttributeError: pass
    ms = (time.perf_counter() - t0) * 1e3
    return ms / (repeats * n_latents)

# Extended seed list: scales runtime from ~8 min (5 seeds) to ~8 h (300 seeds)
seeds = list(range(2, 102))  # 100 distinct seeds
# Will collect FID curves for each seed
avg_results = {'baseline': [], 'boson': [], 'boson_nl': []}

### Helper Functions for FID Calculation ###
def get_activations(images, model, device):
    """
    Extract features from images using the given Inception model.
    Expects images of shape (N, C, H, W) in range [0, 1]. If grayscale, channels are repeated.
    Resizes images to 299x299.
    """
    # If images are grayscale, convert to 3 channels.
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    # Resize images to 299x299 (the Inception input size)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    # Normalize using Inception's normalization parameters
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Apply normalization per image in the batch
    # Note: Since transforms.Normalize works on single images, we apply it manually.
    for i in range(images.shape[0]):
        images[i] = normalize(images[i])
    with torch.no_grad():
        features = model(images.to(device))
    return features

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Compute the Frechet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    # Numerical error might give slight imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return fid

def compute_fid(real_images, generated_images, model, device):
    """
    Compute the FID score between a batch of real images and generated images.
    """
    real_features = get_activations(real_images, model, device)
    gen_features = get_activations(generated_images, model, device)
    real_features = real_features.cpu().numpy()
    gen_features = gen_features.cpu().numpy()
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(gen_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(gen_features, rowvar=False)
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

# Paralellisation

# will hold a singleton generator inside each worker
_WORKER_LATENT_GEN = None
# Separate singleton for the *non‑linear* sampler
_WORKER_LATENT_GEN_NL = None

def _worker_init(gen_state_dict, latent_dim):
    """Runs once in every worker process."""
    global _WORKER_LATENT_GEN
    torch.manual_seed(int.from_bytes(os.urandom(4), "little"))  # decouple RNG streams
    # Re-create BosonLatentGenerator from its state_dict
    bs_torch = BosonSamplerTorch(**gen_state_dict['sampler_kwargs'])
    _WORKER_LATENT_GEN = BosonLatentGenerator(latent_dim, bs_torch)

def _single_latent(_dummy):
    """Return one latent (shape [1, latent_dim]) using the per-worker generator."""
    return _WORKER_LATENT_GEN(1)

# --- Nonlinear worker helpers ---
def _worker_init_nl(gen_state_dict, latent_dim):
    """Runs once in every worker process for the non-linear Boson sampler."""
    global _WORKER_LATENT_GEN_NL
    torch.manual_seed(int.from_bytes(os.urandom(4), "little"))  # decouple RNG streams
    bs_torch_nl = BosonSamplerTorch(**gen_state_dict['sampler_kwargs'])
    _WORKER_LATENT_GEN_NL = BosonLatentGenerator(latent_dim, bs_torch_nl)

def _single_latent_nl(_dummy):
    """Return one latent (shape [1, latent_dim]) from the non-linear generator."""
    return _WORKER_LATENT_GEN_NL(1)

def log_csv(run, seed, phase, epoch,
            baseline_loss, boson_loss, boson_nonlinear_loss,
            baseline_latency_ms="N/A", boson_latency_ms="N/A", boson_nl_latency_ms="N/A",
            csv_file='generator_losses_nonlinearity.csv'):
    """
    Append a row to the CSV file with current generator losses (or "N/A")
    and, optionally, latency totals for each branch.
    """
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            run, seed, phase, epoch,
            baseline_loss, boson_loss, boson_nonlinear_loss,
            baseline_latency_ms, boson_latency_ms, boson_nl_latency_ms,
            datetime.now()
        ])

def main():
    overall_start = time.perf_counter()
    for seed in seeds:
        # Set global seed for this run
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Prepare fresh FID lists for this seed
        fid_boson_list = []
        fid_boson_nl_list = []
        fid_baseline_list = []
        epoch_list = []
        ### HYPERPARAMETERS
        INPUT_STATE = [1,0,1,0,1,0,1,0,1,0,1,0]
        boson_sampler_params = {
            "input_state": INPUT_STATE,
            "tbi_params": {
                "input_loss": 0.0,
                "detector_efficiency": 1,
                "bs_loss": 0,
                "bs_noise": 0,
                "distinguishable": False,
                "n_signal_detectors": 0,
                "g2": 0,
                "tbi_type": "multi-loop",
                "n_loops": 1000,
                #"loop_lengths": [1, 2],
                "postselected": True
            },
            "n_tiling": 1
        }
        DISCRIMINATOR_ITER = 50
        PRETRAIN_DISC = -1
        PRETRAIN_GEN = -1
        NUM_ITER = 1
        DISC_LR = 1e-5
        GEN_LR = 1e-4

        # Folders for data.
        ct_folder = '/Users/olivernorregaard/Downloads/DiscretePETCT-main/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805'
        pet_folder = '/Users/olivernorregaard/Downloads/DiscretePETCT-main/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/1.000000-PET NAC-24000'

        num_workers = 4
        dicom_files = create_data_loader(ct_folder=ct_folder, pet_folder=pet_folder, num_workers=num_workers, augment=False)
        
        class DummyLatentSpace:
            def __init__(self, dim: int):
                self.dim = dim
            def generate(self, batch_size: int):
                return torch.randn(batch_size, self.dim)
        # Create latent generators.
        latent_dim = len(boson_sampler_params["input_state"])
        baseline_latent_space = DummyLatentSpace(latent_dim)
        boson_latent_space_torch = BosonSamplerTorch(
            early_late_pairs=len(INPUT_STATE),
            input_state=INPUT_STATE,
            num_loops=1000,
            input_loss=0,
            coupling_efficiency=1,
            detector_inefficiency=1,
            mu=1,
            temporal_mismatch=0,
            spectral_mismatch=0,
            arrival_time_jitter=0,
            bs_loss=1,
            bs_jitter=0,
            phase_noise_std=0,
            systematic_phase_offset=0,
            mode_loss=np.ones(12),
            dark_count_rate=0,
            use_advanced_nonlinearity=False,
            g2_target=0
        )
        boson_latent_space = BosonLatentGenerator(latent_dim, boson_latent_space_torch)
        boson_latent_space_torch_nonlinear = BosonSamplerTorch(            
            early_late_pairs=len(INPUT_STATE),
            input_state=INPUT_STATE,
            num_loops=1000,
            input_loss=0,
            coupling_efficiency=1,
            detector_inefficiency=1,
            mu=1,
            temporal_mismatch=0,
            spectral_mismatch=0,
            arrival_time_jitter=0,
            bs_loss=1,
            bs_jitter=0,
            phase_noise_std=0,
            systematic_phase_offset=0,
            mode_loss=np.ones(12),
            dark_count_rate=0,
            use_advanced_nonlinearity=True,
            pulse_bw=0.5,
            detuning=0.0,
            phi=0.0,
            g2_target=0,
        )
        boson_latent_space_nonlinear = BosonLatentGenerator(latent_dim, boson_latent_space_torch_nonlinear)
            # ------------------------------------------------------------------
        # Parallel-generation helpers
        # ------------------------------------------------------------------USE_PARALLEL = bool(int(os.getenv("USE_PARALLEL", "0")))
        USE_PARALLEL = bool(int(os.getenv("USE_PARALLEL", "0")))

        # build a snapshot of args to send to workers
        _sdict = {
            'sampler_kwargs': {
                'early_late_pairs': len(INPUT_STATE),
                'input_state': INPUT_STATE,
                'num_loops': 100,
                'input_loss': 0,
                'coupling_efficiency': 1,
                'detector_inefficiency': 1,
                'mu': 1,
                'temporal_mismatch': 0,
                'spectral_mismatch': 0,
                'arrival_time_jitter': 0,
                'bs_loss': 1,
                'bs_jitter': 0,
                'phase_noise_std': 0,
                'systematic_phase_offset': 0,
                'mode_loss': np.ones(12),
                'dark_count_rate': 0,
                'use_advanced_nonlinearity': False,
                'g2_target': 0,
            }
        }
        _sdict_nl = {
            'sampler_kwargs': {
                'early_late_pairs': len(INPUT_STATE),
                'input_state': INPUT_STATE,
                'num_loops': 100,
                'input_loss': 0,
                'coupling_efficiency': 1,
                'detector_inefficiency': 1,
                'mu': 1,
                'temporal_mismatch': 0,
                'spectral_mismatch': 0,
                'arrival_time_jitter': 0,
                'bs_loss': 1,
                'bs_jitter': 0,
                'phase_noise_std': 0,
                'systematic_phase_offset': 0,
                'mode_loss': np.ones(12),
                'dark_count_rate': 0,
                'use_advanced_nonlinearity': True,
                'g2_target': 0,
            }
        }

        if USE_PARALLEL:
            pool = mp.Pool(processes=mp.cpu_count(),
                        initializer=_worker_init,
                        initargs=(_sdict, latent_dim))
            pool_nl = mp.Pool(processes=mp.cpu_count(),
                            initializer=_worker_init_nl,
                            initargs=(_sdict_nl, latent_dim))

        def generate_latents(batch_size):
            if not USE_PARALLEL:
                return boson_latent_space(batch_size)
            # map range(batch_size) just to call the worker n times
            lat_list = pool.map(_single_latent, range(batch_size))
            return torch.cat(lat_list, dim=0)

        get_boson_latent = generate_latents

        def generate_latents_nl(batch_size):
            if not USE_PARALLEL:
                return boson_latent_space_nonlinear(batch_size)
            lat_list = pool_nl.map(_single_latent_nl, range(batch_size))
            return torch.cat(lat_list, dim=0)

        get_boson_latent_nonlinear = generate_latents_nl

        # ------------------------------------------------------------------
        #  Quick micro‑benchmark (runs once at start, negligible cost)
        # ------------------------------------------------------------------
        _bench_bs = 32
        t0 = time.perf_counter(); _ = boson_latent_space(_bench_bs);  serial = time.perf_counter() - t0
        if USE_PARALLEL:
            t0 = time.perf_counter(); _ = generate_latents(_bench_bs); parallel = time.perf_counter() - t0
        else:
            parallel = serial
        print(f"[Benchmark] Latents for batch {_bench_bs}:  "
            f"serial {serial:.3f}s | parallel {parallel:.3f}s "
            f"(cores = {mp.cpu_count()}) | speed-up ≈ {serial/parallel if parallel else 1:.2f}×")
        if USE_PARALLEL:
            t0 = time.perf_counter(); _ = generate_latents_nl(_bench_bs); parallel_nl = time.perf_counter() - t0
            print(f"[Benchmark‑NL] Latents(NL) for batch {_bench_bs}: serial {serial:.3f}s | "
                f"parallel {parallel_nl:.3f}s | speed-up ≈ {serial/parallel_nl if parallel_nl else 1:.2f}×")

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        else:
            device = torch.device("cpu")
            print("Using CPU")

        # Select the correct autocast target for torch.amp.autocast
        autocast_device = 'mps' if device.type == 'mps' else ('cuda' if device.type == 'cuda' else 'cpu')
            
        # Load pretrained InceptionV3 model for FID (remove the final classification layer)
        inception_model = models.inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        inception_model.fc = nn.Identity()
        inception_model.to(device)
        inception_model.eval()

        torch.manual_seed(seed)
        baseline_generator = Generator(latent_dim).to(device)
        torch.manual_seed(seed)
        baseline_discriminator = Discriminator().to(device)

        torch.manual_seed(seed)
        boson_generator = Generator(latent_dim).to(device)
        torch.manual_seed(seed)
        boson_discriminator = Discriminator().to(device)

        torch.manual_seed(seed)
        boson_generator_nonlinear = Generator(latent_dim).to(device)
        torch.manual_seed(seed)
        boson_discriminator_nonlinear = Discriminator().to(device)


        # Load data
        looper = cycle(dicom_files)
        batch = next(looper)
        pet_images, ct_images = batch
        data_real = ct_images.to(device).float()

        # -------------------------------------------------
        # One-time latency probe (runs once per seed)
        # -------------------------------------------------
        baseline_lat_ms = measure_sampler_time(
            baseline_latent_space.generate, n_latents=256, repeats=40, device_str="cpu")
        boson_lat_ms = measure_sampler_time(
            get_boson_latent, n_latents=64, repeats=20, device_str="cpu")
        bosonNL_lat_ms = measure_sampler_time(
            get_boson_latent_nonlinear, n_latents=64, repeats=20, device_str="cpu")

        print(f"[Sampler latency] Gaussian {baseline_lat_ms:.4f} ms | "
              f"Boson {boson_lat_ms:.4f} ms | "
              f"Boson-NL {bosonNL_lat_ms:.4f} ms")

        # cumulative totals
        baseline_time_ms = boson_time_ms = bosonNL_time_ms = 0.0

        pet_images = pet_images.to(device).float()
        batch_size = pet_images.shape[0]
        
        # ---------------------------------------------
        # Fixed evaluation latents to reduce FID variance
        # ---------------------------------------------
        eval_batch_size = batch_size
        with torch.no_grad():
            fixed_baseline_latent = baseline_latent_space.generate(eval_batch_size).to(device)
            fixed_boson_latent = get_boson_latent(eval_batch_size).to(device)
            fixed_boson_nl_latent = get_boson_latent_nonlinear(eval_batch_size).to(device)
        
        # Create optimizers
        baseline_gen_optimizer = optim.Adam(baseline_generator.parameters(), lr=GEN_LR)
        baseline_disc_optimizer = optim.Adam(baseline_discriminator.parameters(), lr=DISC_LR)
        boson_gen_optimizer = optim.Adam(boson_generator.parameters(), lr=GEN_LR)
        boson_disc_optimizer = optim.Adam(boson_discriminator.parameters(), lr=DISC_LR)
        boson_gen_optimizer_nonlinear = optim.Adam(boson_generator_nonlinear.parameters(), lr=GEN_LR)
        boson_disc_optimizer_nonlinear = optim.Adam(boson_discriminator_nonlinear.parameters(), lr=DISC_LR)

        # Learning rate schedulers
        baseline_gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(baseline_gen_optimizer, 'min', factor=0.5, patience=10, verbose=True)
        baseline_disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(baseline_disc_optimizer, 'min', factor=0.5, patience=10, verbose=True)
        boson_gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(boson_gen_optimizer, 'min', factor=0.5, patience=10, verbose=True)
        boson_disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(boson_disc_optimizer, 'min', factor=0.5, patience=10, verbose=True)

        boson_gen_nonlinear_scheduler = optim.lr_scheduler.ReduceLROnPlateau(boson_gen_optimizer_nonlinear, 'min', factor=0.5, patience=10, verbose=True)
        boson_disc_nonlinear_scheduler = optim.lr_scheduler.ReduceLROnPlateau(boson_disc_optimizer_nonlinear, 'min', factor=0.5, patience=10, verbose=True)
        # Mixed Precision Scalers
        baseline_scaler_gen = torch.amp.GradScaler()
        baseline_scaler_disc = torch.amp.GradScaler()
        boson_scaler_gen = torch.amp.GradScaler()
        boson_scaler_disc = torch.amp.GradScaler()
        boson_scaler_gen_nonlinear = torch.amp.GradScaler()
        boson_scaler_disc_nonlinear = torch.amp.GradScaler()
        # ---------------------------
        # CSV Logging Setup
        # ---------------------------
        csv_file = 'generator_losses_nonlinearity.csv'
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'Run', 'Seed', 'Phase', 'Epoch',
                    'Baseline_Loss', 'Boson_Gen_Loss', 'Boson_Gen_Nonlinear_Loss',
                    'Baseline_Latency_ms', 'Boson_Latency_ms', 'Boson_NL_Latency_ms',
                    'Timestamp'
                ])

        # ---------------------------
        # Pre-training: Discriminators (omitted CSV logging here)
        # ---------------------------
        print("Starting Baseline Discriminator Pre-training")
        for epoch in range(PRETRAIN_DISC):
            for p in baseline_discriminator.parameters():
                p.requires_grad_(True)
            for p in baseline_generator.parameters():
                p.requires_grad_(False)
            if epoch + 1 == 25:
                print("Dropping Baseline disc. learning rate")
                baseline_disc_scheduler.base_lr = DISC_LR * 0.1

            with torch.no_grad():
                baseline_latent = baseline_latent_space.generate(batch_size).to(device)
                data_fake = baseline_generator(baseline_latent).detach()
            with torch.amp.autocast(autocast_device):
                real_output = baseline_discriminator(data_real)
                fake_output = baseline_discriminator(data_fake)
                real_target = torch.ones_like(real_output)
                disc_loss_real = dice_loss(real_output, real_target)
                disc_loss = disc_loss_real
            baseline_disc_optimizer.zero_grad()
            baseline_scaler_disc.scale(disc_loss).backward()
            baseline_scaler_disc.step(baseline_disc_optimizer)
            baseline_scaler_disc.update()
            print(f"Baseline Disc Pre-train Epoch {epoch+1}, Loss: {disc_loss.item()}")
            baseline_disc_scheduler.step(disc_loss)

        print("Starting Boson Discriminator Pre-training")
        for epoch in range(PRETRAIN_DISC):
            for p in boson_discriminator.parameters():
                p.requires_grad_(True)
            for p in boson_generator.parameters():
                p.requires_grad_(False)
            if epoch + 1 == 25:
                print("Dropping Boson disc. learning rate")
                boson_disc_scheduler.base_lr = DISC_LR * 0.1

            with torch.no_grad():
                boson_latent = get_boson_latent(batch_size).to(device)
                data_fake = boson_generator(boson_latent).detach()
            with torch.amp.autocast(autocast_device):
                real_output = boson_discriminator(data_real)
                fake_output = boson_discriminator(data_fake)
                real_target = torch.ones_like(real_output)
                disc_loss_real = dice_loss(real_output, real_target)
                disc_loss = disc_loss_real
            boson_disc_optimizer.zero_grad()
            boson_scaler_disc.scale(disc_loss).backward()
            boson_scaler_disc.step(boson_disc_optimizer)
            boson_scaler_disc.update()
            print(f"Boson Disc Pre-train Epoch {epoch+1}, Loss: {disc_loss.item()}")
            boson_disc_scheduler.step(disc_loss)
        
        print("Starting Boson Discriminator Nonlinear Pre-training")
        for epoch in range(PRETRAIN_DISC):
            for p in boson_discriminator_nonlinear.parameters():
                p.requires_grad_(True)
            for p in boson_generator_nonlinear.parameters():
                p.requires_grad_(False)
            if epoch + 1 == 25:
                print("Dropping Boson disc. learning rate")
                boson_disc_nonlinear_scheduler.base_lr = DISC_LR * 0.1

            with torch.no_grad():
                boson_latent = get_boson_latent_nonlinear(batch_size).to(device)
                data_fake = boson_generator_nonlinear(boson_latent).detach()
            with torch.amp.autocast(autocast_device):
                real_output = boson_discriminator_nonlinear(data_real)
                fake_output = boson_discriminator_nonlinear(data_fake)
                real_target = torch.ones_like(real_output)
                disc_loss_real = dice_loss(real_output, real_target)
                disc_loss = disc_loss_real
            boson_disc_optimizer_nonlinear.zero_grad()
            boson_scaler_disc_nonlinear.scale(disc_loss).backward()
            boson_scaler_disc_nonlinear.step(boson_disc_optimizer_nonlinear)
            boson_scaler_disc_nonlinear.update()
            print(f"Boson Disc Pre-train Epoch {epoch+1}, Loss: {disc_loss.item()}")
            boson_disc_nonlinear_scheduler.step(disc_loss)

        # ---------------------------
        # Pre-training: Generators
        # ---------------------------
        print("Starting Baseline Generator Pre-training")
        for epoch in range(PRETRAIN_GEN):
            for p in baseline_discriminator.parameters():
                p.requires_grad_(False)
            for p in baseline_generator.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast(autocast_device):
                baseline_latent = baseline_latent_space.generate(batch_size).to(device)
                generated_ct = baseline_generator(baseline_latent)
                data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                generated_norm = (generated_ct - generated_ct.min()) / (generated_ct.max() - generated_ct.min())
                fake_output_for_gen = baseline_discriminator(generated_norm)
                adversarial_loss = dice_loss(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                baseline_gen_loss = F.l1_loss(generated_norm, data_norm) + 0.5 * adversarial_loss
            baseline_gen_optimizer.zero_grad()
            baseline_scaler_gen.scale(baseline_gen_loss).backward()
            baseline_scaler_gen.step(baseline_gen_optimizer)
            baseline_scaler_gen.update()
            print(f"Baseline Gen Pre-train Epoch {epoch+1}, Loss: {baseline_gen_loss.item()}")
            baseline_gen_scheduler.step(baseline_gen_loss)

            log_csv("Pre-training", seed, "Pre-training_Baseline", epoch + 1,
                    baseline_gen_loss.item(), "N/A", "N/A",
                    csv_file=csv_file)

        # Boson Generator Pre-training
        print("Starting Boson Generator Pre-training")
        for epoch in range(PRETRAIN_GEN):
            for p in boson_discriminator.parameters():
                p.requires_grad_(False)
            for p in boson_generator.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast(autocast_device):
                boson_latent = get_boson_latent(batch_size).to(device)
                generated_ct = boson_generator(boson_latent)
                data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                generated_norm = (generated_ct - generated_ct.min()) / (generated_ct.max() - generated_ct.min())
                fake_output_for_gen = boson_discriminator(generated_norm)
                adversarial_loss = dice_loss(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                boson_gen_loss = F.l1_loss(generated_norm, data_norm) + 0.5 * adversarial_loss
            boson_gen_optimizer.zero_grad()
            boson_scaler_gen.scale(boson_gen_loss).backward()
            boson_scaler_gen.step(boson_gen_optimizer)
            boson_scaler_gen.update()
            print(f"Boson Gen Pre-train Epoch {epoch+1}, Loss: {boson_gen_loss.item()}")
            boson_gen_scheduler.step(boson_gen_loss)

            log_csv("Pre-training", seed, "Pre-training_Boson", epoch + 1,
                    "N/A", boson_gen_loss.item(), "N/A",
                    csv_file=csv_file)
        
        # Boson Nonlinear Generator Pre-training
        print("Starting Boson Generator Nonlinear Pre-training")
        for epoch in range(PRETRAIN_GEN):
            for p in boson_discriminator_nonlinear.parameters():
                p.requires_grad_(False)
            for p in boson_generator_nonlinear.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast(autocast_device):
                boson_latent_nonlinear = get_boson_latent_nonlinear(batch_size).to(device)  # Ensure correct latent space for nonlinear
                generated_ct_nonlinear = boson_generator_nonlinear(boson_latent_nonlinear)
                data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                generated_norm_nonlinear = (generated_ct_nonlinear - generated_ct_nonlinear.min()) / (generated_ct_nonlinear.max() - generated_ct_nonlinear.min())
                fake_output_for_gen_nonlinear = boson_discriminator_nonlinear(generated_norm_nonlinear)
                adversarial_loss_nonlinear = dice_loss(fake_output_for_gen_nonlinear, torch.ones_like(fake_output_for_gen_nonlinear))
                boson_gen_loss_nonlinear = F.l1_loss(generated_norm_nonlinear, data_norm) + 0.5 * adversarial_loss_nonlinear
            boson_gen_optimizer_nonlinear.zero_grad()
            boson_scaler_gen_nonlinear.scale(boson_gen_loss_nonlinear).backward()
            boson_scaler_gen_nonlinear.step(boson_gen_optimizer_nonlinear)
            boson_scaler_gen_nonlinear.update()
            print(f"Boson Nonlinear Gen Pre-train Epoch {epoch+1}, Loss: {boson_gen_loss_nonlinear.item()}")
            boson_gen_nonlinear_scheduler.step(boson_gen_loss_nonlinear)

            log_csv("Pre-training", seed, "Pre-training_Boson_Nonlinear", epoch + 1,
                    "N/A", "N/A", boson_gen_loss_nonlinear.item(),
                    csv_file=csv_file)
        

        # FID lists already initialized per seed above
        log_csv("Pre-training", seed, "Pre-training_End", "N/A",
                "N/A", "N/A", "N/A",
                csv_file=csv_file)
        # ---------------------------
        # Main Training Loop
        # ---------------------------
        print("Starting main training")
        for run in range(NUM_ITER):
            for epoch in range(DISCRIMINATOR_ITER):
                # -------------------------------------------------
                # Pull a *new* real CT/PET pair for this iteration
                # -------------------------------------------------
                pet_images, ct_images = next(looper)
                data_real = ct_images.to(device).float()
                batch_size = pet_images.shape[0]  # ensure latent dims match
                # --- Baseline Branch Training (Discriminator update) ---
                for p in baseline_discriminator.parameters():
                    p.requires_grad_(True)
                for p in baseline_generator.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    baseline_latent = baseline_latent_space.generate(batch_size).to(device)
                    t_train = time.perf_counter()
                    baseline_data_fake = baseline_generator(baseline_latent).detach()
                with torch.amp.autocast(autocast_device):
                    baseline_real_output = baseline_discriminator(data_real)
                    baseline_fake_output = baseline_discriminator(baseline_data_fake)
                    baseline_real_target = torch.ones_like(baseline_real_output)
                    baseline_fake_target = torch.zeros_like(baseline_fake_output)
                    baseline_disc_loss_real = dice_loss(baseline_real_output, baseline_real_target)
                    baseline_disc_loss_fake = dice_loss(baseline_fake_output, baseline_fake_target)
                    baseline_disc_loss = baseline_disc_loss_real + baseline_disc_loss_fake
                baseline_disc_optimizer.zero_grad()
                baseline_scaler_disc.scale(baseline_disc_loss).backward()
                baseline_scaler_disc.step(baseline_disc_optimizer)
                baseline_scaler_disc.update()
                baseline_time_ms += (time.perf_counter() - t_train) * 1e3

                # --- Boson Branch Training (Discriminator update) ---
                for p in boson_discriminator.parameters():
                    p.requires_grad_(True)
                for p in boson_generator.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    boson_latent = get_boson_latent(batch_size).to(device)
                    t_train = time.perf_counter()
                    boson_data_fake = boson_generator(boson_latent).detach()
                with torch.amp.autocast(autocast_device):
                    boson_real_output = boson_discriminator(data_real)
                    boson_fake_output = boson_discriminator(boson_data_fake)
                    boson_real_target = torch.ones_like(boson_real_output)
                    boson_fake_target = torch.zeros_like(boson_fake_output)
                    boson_disc_loss_real = dice_loss(boson_real_output, boson_real_target)
                    boson_disc_loss_fake = dice_loss(boson_fake_output, boson_fake_target)
                    boson_disc_loss = boson_disc_loss_real + boson_disc_loss_fake
                boson_disc_optimizer.zero_grad()
                boson_scaler_disc.scale(boson_disc_loss).backward()
                boson_scaler_disc.step(boson_disc_optimizer)
                boson_scaler_disc.update()
                boson_time_ms += (time.perf_counter() - t_train) * 1e3

                # --- Boson Nonlinear Branch Training (Discriminator update) ---
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(True)
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    boson_latent_nonlinear = get_boson_latent_nonlinear(batch_size).to(device)
                    t_train = time.perf_counter()
                    boson_data_fake_nonlinear = boson_generator_nonlinear(boson_latent_nonlinear).detach()
                with torch.amp.autocast(autocast_device):
                    boson_real_output_nonlinear = boson_discriminator_nonlinear(data_real)
                    boson_fake_output_nonlinear = boson_discriminator_nonlinear(boson_data_fake_nonlinear)
                    boson_real_target_nonlinear = torch.ones_like(boson_real_output_nonlinear)
                    boson_fake_target_nonlinear = torch.zeros_like(boson_fake_output_nonlinear)
                    boson_disc_loss_real_nonlinear = dice_loss(boson_real_output_nonlinear, boson_real_target_nonlinear)
                    boson_disc_loss_fake_nonlinear = dice_loss(boson_fake_output_nonlinear, boson_fake_target_nonlinear)
                    boson_disc_loss_nonlinear = boson_disc_loss_real_nonlinear + boson_disc_loss_fake_nonlinear
                boson_disc_optimizer_nonlinear.zero_grad()
                boson_scaler_disc_nonlinear.scale(boson_disc_loss_nonlinear).backward()
                boson_scaler_disc_nonlinear.step(boson_disc_optimizer_nonlinear)
                boson_scaler_disc_nonlinear.update()
                bosonNL_time_ms += (time.perf_counter() - t_train) * 1e3


                # --- Generator Training ---
                # Freeze all discriminators for Baseline generator update
                for p in baseline_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(False)
                # Unfreeze Baseline generator only
                for p in baseline_generator.parameters():
                    p.requires_grad_(True)
                # Ensure other generators remain frozen
                for p in boson_generator.parameters():
                    p.requires_grad_(False)
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(False)
                # Baseline's simulator
                with torch.amp.autocast(autocast_device):
                    baseline_latent = baseline_latent_space.generate(batch_size).to(device)
                    t_train = time.perf_counter()
                    baseline_generated_ct = baseline_generator(baseline_latent)
                    data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                    baseline_generated_norm = (baseline_generated_ct - baseline_generated_ct.min()) / (baseline_generated_ct.max() - baseline_generated_ct.min())
                    baseline_fake_output_for_gen = baseline_discriminator(baseline_generated_norm)
                    baseline_adversarial_loss = dice_loss(baseline_fake_output_for_gen, torch.ones_like(baseline_fake_output_for_gen))
                    baseline_gen_loss = F.l1_loss(baseline_generated_norm, data_norm) + 0.5 * baseline_adversarial_loss
                baseline_gen_optimizer.zero_grad()
                baseline_scaler_gen.scale(baseline_gen_loss).backward()
                baseline_scaler_gen.step(baseline_gen_optimizer)
                baseline_scaler_gen.update()
                baseline_time_ms += (time.perf_counter() - t_train) * 1e3

                # Freeze all discriminators for Boson generator update
                for p in boson_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(False)
                # Unfreeze Boson generator only
                for p in boson_generator.parameters():
                    p.requires_grad_(True)
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(False)
                # Our simulator
                with torch.amp.autocast(autocast_device):
                    boson_latent = get_boson_latent(batch_size).to(device)
                    t_train = time.perf_counter()
                    boson_generated_ct = boson_generator(boson_latent)
                    data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                    boson_generated_norm = (boson_generated_ct - boson_generated_ct.min()) / (boson_generated_ct.max() - boson_generated_ct.min())
                    boson_fake_output_for_gen = boson_discriminator(boson_generated_norm)
                    boson_adversarial_loss = dice_loss(boson_fake_output_for_gen, torch.ones_like(boson_fake_output_for_gen))
                    boson_gen_loss = F.l1_loss(boson_generated_norm, data_norm) + 0.5 * boson_adversarial_loss
                boson_gen_optimizer.zero_grad()
                boson_scaler_gen.scale(boson_gen_loss).backward()
                boson_scaler_gen.step(boson_gen_optimizer)
                boson_scaler_gen.update()
                boson_time_ms += (time.perf_counter() - t_train) * 1e3

                # Freeze all discriminators for Boson Nonlinear generator update
                for p in boson_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(False)
                # Unfreeze Boson Nonlinear generator only
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(True)
                # Ensure other generators remain frozen
                for p in boson_generator.parameters():
                    p.requires_grad_(False)
                # Boson Nonlinear Generator
                with torch.amp.autocast(autocast_device):
                    boson_latent_nonlinear = get_boson_latent_nonlinear(batch_size).to(device)
                    t_train = time.perf_counter()
                    boson_generated_ct_nonlinear = boson_generator_nonlinear(boson_latent_nonlinear)
                    data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                    boson_generated_norm_nonlinear = (boson_generated_ct_nonlinear - boson_generated_ct_nonlinear.min()) / (boson_generated_ct_nonlinear.max() - boson_generated_ct_nonlinear.min())
                    boson_fake_output_for_gen_nonlinear = boson_discriminator_nonlinear(boson_generated_norm_nonlinear)
                    boson_adversarial_loss_nonlinear = dice_loss(boson_fake_output_for_gen_nonlinear, torch.ones_like(boson_fake_output_for_gen_nonlinear))
                    boson_gen_loss_nonlinear = F.l1_loss(boson_generated_norm_nonlinear, data_norm) + 0.5 * boson_adversarial_loss_nonlinear
                boson_gen_optimizer_nonlinear.zero_grad()
                boson_scaler_gen_nonlinear.scale(boson_gen_loss_nonlinear).backward()
                boson_scaler_gen_nonlinear.step(boson_gen_optimizer_nonlinear)
                boson_scaler_gen_nonlinear.update()
                bosonNL_time_ms += (time.perf_counter() - t_train) * 1e3

                log_csv(run + 1, seed, "Main Training", epoch + 1,
                        baseline_gen_loss.item(),
                        boson_gen_loss.item(),
                        boson_gen_loss_nonlinear.item(),
                        f"{baseline_time_ms:.2f}",
                        f"{boson_time_ms:.2f}",
                        f"{bosonNL_time_ms:.2f}",
                        csv_file)

                print(f"Epoch {epoch+1}, Baseline Gen Loss: {baseline_gen_loss.item()}, Baseline Disc Loss: {baseline_disc_loss.item()}, " +
                    f"Boson Gen Loss: {boson_gen_loss.item()}, Boson Disc Loss: {boson_disc_loss.item()}" +
                    f"Boson Gen Nonlinear Loss: {boson_gen_loss_nonlinear.item()}, Boson Disc Loss: {boson_disc_loss_nonlinear.item()}")

                if (epoch + 1) % 250 == 0:
                    model_save_folder = 'model_checkpoints'
                    os.makedirs(model_save_folder, exist_ok=True)
                    baseline_generator_save_path = os.path.join(model_save_folder, f'baseline_generator_epoch_{epoch + 1}.pt')
                    baseline_discriminator_save_path = os.path.join(model_save_folder, f'baseline_discriminator_epoch_{epoch + 1}.pt')
                    boson_generator_save_path = os.path.join(model_save_folder, f'boson_generator_epoch_{epoch + 1}.pt')
                    boson_discriminator_save_path = os.path.join(model_save_folder, f'boson_discriminator_epoch_{epoch + 1}.pt')
                    boson_generator_nonlinear_save_path = os.path.join(model_save_folder, f'boson_nonlinear_generator_epoch_{epoch + 1}.pt')
                    boson_discriminator_nonlinear_save_path = os.path.join(model_save_folder, f'boson_nonlinear_discriminator_epoch_{epoch + 1}.pt')
                    
                    save_weights(baseline_generator, baseline_gen_optimizer, epoch + 1, baseline_generator_save_path, baseline_gen_loss.item())
                    save_weights(baseline_discriminator, baseline_disc_optimizer, epoch + 1, baseline_discriminator_save_path, baseline_disc_loss.item())
                    save_weights(boson_generator, boson_gen_optimizer, epoch + 1, boson_generator_save_path, boson_gen_loss.item())
                    save_weights(boson_discriminator, boson_disc_optimizer, epoch + 1, boson_discriminator_save_path, boson_disc_loss.item())
                    save_weights(boson_generator, boson_gen_optimizer_nonlinear, epoch + 1, boson_generator_nonlinear_save_path, boson_gen_loss_nonlinear.item())
                    save_weights(boson_discriminator, boson_disc_optimizer_nonlinear, epoch + 1, boson_discriminator_nonlinear_save_path, boson_disc_loss_nonlinear.item())
                if (epoch + 1) % 50 == 0:
                    with torch.no_grad():
                        # Use fixed latents for evaluation to reduce variance
                        baseline_generated = baseline_generator(fixed_baseline_latent)
                        boson_generated = boson_generator(fixed_boson_latent)
                        boson_generated_nl = boson_generator_nonlinear(fixed_boson_nl_latent)
                        # Compute FID scores using the same real images batch
                        fid_baseline = compute_fid(data_real, baseline_generated, inception_model, device)
                        fid_boson = compute_fid(data_real, boson_generated, inception_model, device)
                        fid_boson_nl = compute_fid(data_real, boson_generated_nl, inception_model, device)
                    fid_baseline_list.append(fid_baseline)
                    fid_boson_list.append(fid_boson)
                    fid_boson_nl_list.append(fid_boson_nl)
                    epoch_list.append(epoch+1)
                    print(f"Epoch {epoch+1} FID - Baseline: {fid_baseline:.3f}, Boson: {fid_boson:.3f}, Boson NL: {fid_boson_nl:.3f}")
                baseline_gen_scheduler.step(baseline_gen_loss)
                baseline_disc_scheduler.step(baseline_disc_loss)
                boson_gen_scheduler.step(boson_gen_loss)
                boson_disc_scheduler.step(boson_disc_loss)
                boson_gen_nonlinear_scheduler.step(boson_gen_loss_nonlinear)
                boson_disc_nonlinear_scheduler.step(boson_disc_loss_nonlinear)
                print(f'Run {run+1}, Baseline Gen Loss: {baseline_gen_loss.item()}, Baseline Disc Loss: {baseline_disc_loss.item()}, ' +
                f'Boson Gen Loss: {boson_gen_loss.item()}, Boson Disc Loss: {boson_disc_loss.item()}' +
                f'Boson Gen Loss: {boson_gen_loss_nonlinear.item()}, Boson Disc Loss: {boson_disc_loss_nonlinear.item()}')
            # Store this seed's FID curve
            avg_results['baseline'].append(fid_baseline_list)
            avg_results['boson'].append(fid_boson_list)
            avg_results['boson_nl'].append(fid_boson_nl_list)
            log_csv("Ops-Summary", seed, "Totals", "N/A",
                    "N/A", "N/A", "N/A",
                    f"{baseline_time_ms:.2f}",
                    f"{boson_time_ms:.2f}",
                    f"{bosonNL_time_ms:.2f}",
                    csv_file)
    # Compute mean FID across all seeds for each epoch
    mean_fid_baseline = np.mean(avg_results['baseline'], axis=0)
    mean_fid_boson = np.mean(avg_results['boson'], axis=0)
    mean_fid_boson_nl = np.mean(avg_results['boson_nl'], axis=0)
    # Plot mean FID curves
    plt.figure()
    plt.plot(epoch_list, mean_fid_baseline, label='Baseline Mean FID')
    plt.plot(epoch_list, mean_fid_boson, label='Boson Mean FID')
    plt.plot(epoch_list, mean_fid_boson_nl, label='Boson NL Mean FID')
    plt.xlabel('Epoch')
    plt.ylabel('Mean FID Score')
    plt.legend()
    plt.title('Mean FID Score vs Epoch Across Seeds')
    plt.show()
    print(f"\n>>> Total runtime with USE_PARALLEL={USE_PARALLEL}: "
        f"{time.perf_counter() - overall_start:.2f} seconds")    
    if USE_PARALLEL:
        pool.close()
        pool_nl.close()
        pool.join()
        pool_nl.join()

if __name__ == '__main__':
    main()