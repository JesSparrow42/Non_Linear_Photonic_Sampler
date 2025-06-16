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
from ptseries.models import PTGenerator
from ptseries.algorithms.gans.utils import infiniteloop
from utils.utils import *
import numpy as np
from bosonsampler import BosonLatentGenerator, BosonSamplerTorch
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import torchvision.transforms as transforms
import torchvision.models as models
import time
import multiprocessing as mp


# Seeds to use for statistical significance testing
seeds = [2, 3, 5, 7, 11]
# Will collect FID curves for each seed
avg_results = {'pt': [], 'boson': [], 'boson_nl': [], 'gauss': []}

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
    """Runs once in every worker process for the *non‑linear* Boson sampler."""
    global _WORKER_LATENT_GEN_NL
    torch.manual_seed(int.from_bytes(os.urandom(4), "little"))  # decouple RNG streams
    bs_torch_nl = BosonSamplerTorch(**gen_state_dict['sampler_kwargs'])
    _WORKER_LATENT_GEN_NL = BosonLatentGenerator(latent_dim, bs_torch_nl)

def _single_latent_nl(_dummy):
    """Return one latent (shape [1, latent_dim]) from the *non‑linear* generator."""
    return _WORKER_LATENT_GEN_NL(1)

def log_csv(run, seed, phase, epoch, pt_loss, boson_loss, boson_nonlinear_loss, gaussian_loss, csv_file):
    """Append a row to the CSV file with current generator losses."""
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            run, seed, phase, epoch, pt_loss, boson_loss, boson_nonlinear_loss, gaussian_loss, datetime.now()
        ])

def main():
    overall_start = time.perf_counter()
    for seed in seeds:
        # Set global seed for this run
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Prepare fresh FID lists for this seed
        fid_pt_list = []
        fid_boson_list = []
        fid_boson_nl_list = []
        fid_gauss_list = []
        epoch_list = []
        ### HYPERPARAMETERS
        INPUT_STATE = [1,0,1,0,1,0,1,0,1,0]
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
        DISCRIMINATOR_ITER = 1500
        PRETRAIN_GEN = 50
        NUM_ITER = 1
        DISC_LR = 1e-5
        GEN_LR = 1e-4

        # Folders for data.
        ct_folder = '/Users/olivernorregaard/Downloads/DiscretePETCT-main/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805'
        pet_folder = '/Users/olivernorregaard/Downloads/DiscretePETCT-main/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/1.000000-PET NAC-24000'

        num_workers = 4
        dicom_files = create_data_loader(ct_folder=ct_folder, pet_folder=pet_folder, num_workers=num_workers, augment=False)
        
        # Create latent generators.
        pt_latent_space = PTGenerator(**boson_sampler_params)
        latent_dim = len(boson_sampler_params["input_state"])
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
            mode_loss=np.ones(len(INPUT_STATE)),
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
            mode_loss=np.ones(len(INPUT_STATE)),
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
                'mode_loss': np.ones(len(INPUT_STATE)),
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
                'mode_loss': np.ones(len(INPUT_STATE)),
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
            
        # Load pretrained InceptionV3 model for FID (remove the final classification layer)
        inception_model = models.inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        inception_model.fc = nn.Identity()
        inception_model.to(device)
        inception_model.eval()

        torch.manual_seed(seed)
        pt_generator = Generator(latent_dim).to(device)
        torch.manual_seed(seed)
        pt_discriminator = Discriminator().to(device)

        torch.manual_seed(seed)
        boson_generator = Generator(latent_dim).to(device)
        torch.manual_seed(seed)
        boson_discriminator = Discriminator().to(device)

        torch.manual_seed(seed)
        boson_generator_nonlinear = Generator(latent_dim).to(device)
        torch.manual_seed(seed)
        boson_discriminator_nonlinear = Discriminator().to(device)

        torch.manual_seed(seed)
        gaussian_generator = Generator(latent_dim).to(device)
        torch.manual_seed(seed)
        gaussian_discriminator = Discriminator().to(device)

        # Load data
        looper = infiniteloop(dicom_files)
        batch = next(looper)
        pet_images, ct_images = batch
        data_real = ct_images.to(device).float()
        pet_images = pet_images.to(device).float()
        batch_size = pet_images.shape[0]
        
        # ---------------------------------------------
        # Fixed evaluation latents to reduce FID variance
        # ---------------------------------------------
        eval_batch_size = batch_size
        with torch.no_grad():
            fixed_pt_latent = pt_latent_space.generate(eval_batch_size).to(device)
            fixed_boson_latent = get_boson_latent(eval_batch_size).to(device)
            fixed_boson_nl_latent = get_boson_latent_nonlinear(eval_batch_size).to(device)
            fixed_gauss_latent = torch.randn(eval_batch_size, latent_dim).to(device)
        
        # Create optimizers
        pt_gen_optimizer = optim.Adam(pt_generator.parameters(), lr=GEN_LR)
        pt_disc_optimizer = optim.Adam(pt_discriminator.parameters(), lr=DISC_LR)
        boson_gen_optimizer = optim.Adam(boson_generator.parameters(), lr=GEN_LR)
        boson_disc_optimizer = optim.Adam(boson_discriminator.parameters(), lr=DISC_LR)
        boson_gen_optimizer_nonlinear = optim.Adam(boson_generator_nonlinear.parameters(), lr=GEN_LR)
        boson_disc_optimizer_nonlinear = optim.Adam(boson_discriminator_nonlinear.parameters(), lr=DISC_LR)
        gaussian_gen_optimizer = optim.Adam(gaussian_generator.parameters(), lr=GEN_LR)
        gaussian_disc_optimizer = optim.Adam(gaussian_discriminator.parameters(), lr=DISC_LR)

        # Learning rate schedulers
        pt_gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pt_gen_optimizer, 'min', factor=0.5, patience=10, verbose=True)
        pt_disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pt_disc_optimizer, 'min', factor=0.5, patience=10, verbose=True)
        boson_gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(boson_gen_optimizer, 'min', factor=0.5, patience=10, verbose=True)
        boson_disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(boson_disc_optimizer, 'min', factor=0.5, patience=10, verbose=True)
        gauss_gen_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gaussian_gen_optimizer, 'min', factor=0.5, patience=10, verbose=True)
        gauss_disc_scheduler = optim.lr_scheduler.ReduceLROnPlateau(gaussian_disc_optimizer, 'min', factor=0.5, patience=10, verbose=True)

        boson_gen_nonlinear_scheduler = optim.lr_scheduler.ReduceLROnPlateau(boson_gen_optimizer_nonlinear, 'min', factor=0.5, patience=10, verbose=True)
        boson_disc_nonlinear_scheduler = optim.lr_scheduler.ReduceLROnPlateau(boson_disc_optimizer_nonlinear, 'min', factor=0.5, patience=10, verbose=True)
        # Mixed Precision Scalers
        pt_scaler_gen = torch.amp.GradScaler()
        pt_scaler_disc = torch.amp.GradScaler()
        boson_scaler_gen = torch.amp.GradScaler()
        boson_scaler_disc = torch.amp.GradScaler()
        boson_scaler_gen_nonlinear = torch.amp.GradScaler()
        boson_scaler_disc_nonlinear = torch.amp.GradScaler()
        gauss_scaler_gen = torch.amp.GradScaler()
        gauss_scaler_disc = torch.amp.GradScaler()
        # ---------------------------
        # CSV Logging Setup
        # ---------------------------
        csv_file = 'generator_losses_nonlinearity.csv'
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Run', 'Seed', 'Phase', 'Epoch', 'PT_Gen_Loss', 'Boson_Gen_Loss', 'Boson_Gen_Nonlinear_Loss', 'Gaussian', 'Timestamp'])

        # ---------------------------
        # Pre-training: Discriminators (omitted CSV logging here)
        # ---------------------------
        print("Starting PT Discriminator Pre-training")
        for epoch in range(50):
            for p in pt_discriminator.parameters():
                p.requires_grad_(True)
            for p in pt_generator.parameters():
                p.requires_grad_(False)
            if epoch + 1 == 25:
                print("Dropping PT disc. learning rate")
                pt_disc_scheduler.base_lr = DISC_LR * 0.1

            with torch.no_grad():
                pt_latent = pt_latent_space.generate(batch_size).to(device)
                data_fake = pt_generator(pt_latent).detach()
            with torch.amp.autocast('mps'):
                real_output = pt_discriminator(data_real)
                fake_output = pt_discriminator(data_fake)
                real_target = torch.ones_like(real_output)
                disc_loss_real = dice_loss(real_output, real_target)
                disc_loss = disc_loss_real
            pt_disc_optimizer.zero_grad()
            pt_scaler_disc.scale(disc_loss).backward()
            pt_scaler_disc.step(pt_disc_optimizer)
            pt_scaler_disc.update()
            print(f"PT Disc Pre-train Epoch {epoch+1}, Loss: {disc_loss.item()}")
            pt_disc_scheduler.step(disc_loss)

        print("Starting Boson Discriminator Pre-training")
        for epoch in range(50):
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
            with torch.amp.autocast('mps'):
                real_output = boson_discriminator(data_real)
                fake_output = boson_discriminator(data_fake)
                real_target = torch.ones_like(real_output)
                disc_loss_real = dice_loss(real_output, real_target)
                disc_loss = disc_loss_real
            boson_disc_optimizer.zero_grad()
            boson_scaler_disc.scale(disc_loss).backward()
            boson_disc_optimizer.step()
            boson_scaler_disc.update()
            print(f"Boson Disc Pre-train Epoch {epoch+1}, Loss: {disc_loss.item()}")
            boson_disc_scheduler.step(disc_loss)
        
        print("Starting Boson Discriminator Nonlinear Pre-training")
        for epoch in range(50):
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
            with torch.amp.autocast('mps'):
                real_output = boson_discriminator_nonlinear(data_real)
                fake_output = boson_discriminator_nonlinear(data_fake)
                real_target = torch.ones_like(real_output)
                disc_loss_real = dice_loss(real_output, real_target)
                disc_loss = disc_loss_real
            boson_disc_optimizer_nonlinear.zero_grad()
            boson_scaler_disc_nonlinear.scale(disc_loss).backward()
            boson_disc_optimizer_nonlinear.step()
            boson_scaler_disc_nonlinear.update()
            print(f"Boson Disc Pre-train Epoch {epoch+1}, Loss: {disc_loss.item()}")
            boson_disc_nonlinear_scheduler.step(disc_loss)

        # ---------------------------
        # Pre-training: Generators
        # ---------------------------
        print("Starting PT Generator Pre-training")
        for epoch in range(PRETRAIN_GEN):
            for p in pt_discriminator.parameters():
                p.requires_grad_(False)
            for p in pt_generator.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast('mps'):
                pt_latent = pt_latent_space.generate(batch_size).to(device)
                generated_ct = pt_generator(pt_latent)
                data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                generated_norm = (generated_ct - generated_ct.min()) / (generated_ct.max() - generated_ct.min())
                fake_output_for_gen = pt_discriminator(generated_norm)
                adversarial_loss = dice_loss(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                pt_gen_loss = F.l1_loss(generated_norm, data_norm) + 0.5 * adversarial_loss
            pt_gen_optimizer.zero_grad()
            pt_scaler_gen.scale(pt_gen_loss).backward()
            pt_scaler_gen.step(pt_gen_optimizer)
            pt_scaler_gen.update()
            print(f"PT Gen Pre-train Epoch {epoch+1}, Loss: {pt_gen_loss.item()}")
            pt_gen_scheduler.step(pt_gen_loss)

            log_csv("Pre-training", seed, "Pre-training_PT", epoch+1, pt_gen_loss.item(), "N/A", "N/A","N/A", csv_file)

        # Boson Generator Pre-training
        print("Starting Boson Generator Pre-training")
        for epoch in range(PRETRAIN_GEN):
            for p in boson_discriminator.parameters():
                p.requires_grad_(False)
            for p in boson_generator.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast('mps'):
                boson_latent = get_boson_latent(batch_size).to(device)
                generated_ct = boson_generator(boson_latent)
                data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                generated_norm = (generated_ct - generated_ct.min()) / (generated_ct.max() - generated_ct.min())
                fake_output_for_gen = boson_discriminator(generated_norm)
                adversarial_loss = dice_loss(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                boson_gen_loss = F.l1_loss(generated_norm, data_norm) + 0.5 * adversarial_loss
            boson_gen_optimizer.zero_grad()
            boson_scaler_gen.scale(boson_gen_loss).backward()
            boson_gen_optimizer.step()
            boson_scaler_gen.step(boson_gen_optimizer)
            boson_scaler_gen.update()
            print(f"Boson Gen Pre-train Epoch {epoch+1}, Loss: {boson_gen_loss.item()}")
            boson_gen_scheduler.step(boson_gen_loss)

            log_csv("Pre-training", seed, "Pre-training_Boson", epoch+1, "N/A", boson_gen_loss.item(), "N/A","N/A", csv_file)
        
        # Boson Nonlinear Generator Pre-training
        print("Starting Boson Generator Nonlinear Pre-training")
        for epoch in range(PRETRAIN_GEN):
            for p in boson_discriminator_nonlinear.parameters():
                p.requires_grad_(False)
            for p in boson_generator_nonlinear.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast('mps'):
                boson_latent_nonlinear = get_boson_latent_nonlinear(batch_size).to(device)  # Ensure correct latent space for nonlinear
                generated_ct_nonlinear = boson_generator_nonlinear(boson_latent_nonlinear)
                data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                generated_norm_nonlinear = (generated_ct_nonlinear - generated_ct_nonlinear.min()) / (generated_ct_nonlinear.max() - generated_ct_nonlinear.min())
                fake_output_for_gen_nonlinear = boson_discriminator_nonlinear(generated_norm_nonlinear)
                adversarial_loss_nonlinear = dice_loss(fake_output_for_gen_nonlinear, torch.ones_like(fake_output_for_gen_nonlinear))
                boson_gen_loss_nonlinear = F.l1_loss(generated_norm_nonlinear, data_norm) + 0.5 * adversarial_loss_nonlinear
            boson_gen_optimizer_nonlinear.zero_grad()
            boson_scaler_gen_nonlinear.scale(boson_gen_loss_nonlinear).backward()
            boson_gen_optimizer_nonlinear.step()
            boson_scaler_gen_nonlinear.step(boson_gen_optimizer_nonlinear)
            boson_scaler_gen_nonlinear.update()
            print(f"Boson Nonlinear Gen Pre-train Epoch {epoch+1}, Loss: {boson_gen_loss_nonlinear.item()}")
            boson_gen_nonlinear_scheduler.step(boson_gen_loss_nonlinear)

            log_csv("Pre-training", seed, "Pre-training_Boson_Nonlinear", epoch+1, "N/A", "N/A", boson_gen_loss_nonlinear.item(), "N/A",csv_file)
        
        print("Starting Gaussian Discriminator Pre-training")
        for epoch in range(50):
            # Enable training for the Gaussian discriminator and freeze its generator.
            for p in gaussian_discriminator.parameters():
                p.requires_grad_(True)
            for p in gaussian_generator.parameters():
                p.requires_grad_(False)
            if epoch + 1 == 25:
                print("Dropping Gaussian disc. learning rate")
                gauss_disc_scheduler.base_lr = DISC_LR * 0.1

            with torch.no_grad():
                # Sample a latent vector from a Gaussian distribution.
                gaussian_latent = torch.randn(batch_size, latent_dim).to(device)
                data_fake = gaussian_generator(gaussian_latent).detach()
            with torch.amp.autocast('mps'):
                real_output = gaussian_discriminator(data_real)
                fake_output = gaussian_discriminator(data_fake)
                real_target = torch.ones_like(real_output)
                disc_loss_real = dice_loss(real_output, real_target)
                # Here we use only the loss on real data for pre-training (as in the examples)
                disc_loss = disc_loss_real
            gaussian_disc_optimizer.zero_grad()
            gauss_scaler_disc.scale(disc_loss).backward()
            gauss_scaler_disc.step(gaussian_disc_optimizer)
            gauss_scaler_disc.update()
            print(f"Gaussian Disc Pre-train Epoch {epoch+1}, Loss: {disc_loss.item()}")
            gauss_disc_scheduler.step(disc_loss)

        print("Starting Gaussian Pre-training")
        for epoch in range(PRETRAIN_GEN):
            for p in gaussian_discriminator.parameters():
                p.requires_grad_(False)
            for p in gaussian_generator.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast('mps'):
                gaussian_latent = torch.randn(batch_size, latent_dim).to(device)
                generated_ct = gaussian_generator(gaussian_latent)
                data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                generated_norm = (generated_ct - generated_ct.min()) / (generated_ct.max() - generated_ct.min())
                fake_output_for_gen = gaussian_discriminator(generated_norm)
                adversarial_loss = dice_loss(fake_output_for_gen, torch.ones_like(fake_output_for_gen))
                gaussian_gen_loss = F.l1_loss(generated_norm, data_norm) + 0.5 * adversarial_loss
            gaussian_gen_optimizer.zero_grad()
            gauss_scaler_gen.scale(gaussian_gen_loss).backward()
            gauss_scaler_gen.step(gaussian_gen_optimizer)
            gauss_scaler_gen.update()
            print(f"Gaussian Gen Pre-train Epoch {epoch+1}, Loss: {gaussian_gen_loss.item()}")
            gauss_gen_scheduler.step(gaussian_gen_loss)

            log_csv("Pre-training", seed, "Pre-training_Gaussan", epoch+1, "N/A", "N/A", "N/A", gaussian_gen_loss.item(), csv_file)

        # FID lists already initialized per seed above
        log_csv("Pre-training", seed, "Pre-training_End", "N/A", "---", "---", "---", "---", csv_file)
        # ---------------------------
        # Main Training Loop
        # ---------------------------
        print("Starting main training")
        for run in range(NUM_ITER):
            for epoch in range(DISCRIMINATOR_ITER):
                # --- PT Branch Training (Discriminator update) ---
                for p in pt_discriminator.parameters():
                    p.requires_grad_(True)
                for p in pt_generator.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    pt_latent = pt_latent_space.generate(batch_size).to(device)
                    pt_data_fake = pt_generator(pt_latent).detach()
                with torch.amp.autocast('mps'):
                    pt_real_output = pt_discriminator(data_real)
                    pt_fake_output = pt_discriminator(pt_data_fake)
                    pt_real_target = torch.ones_like(pt_real_output)
                    pt_fake_target = torch.zeros_like(pt_fake_output)
                    pt_disc_loss_real = dice_loss(pt_real_output, pt_real_target)
                    pt_disc_loss_fake = dice_loss(pt_fake_output, pt_fake_target)
                    pt_disc_loss = pt_disc_loss_real + pt_disc_loss_fake
                pt_disc_optimizer.zero_grad()
                pt_scaler_disc.scale(pt_disc_loss).backward()
                pt_scaler_disc.step(pt_disc_optimizer)
                pt_scaler_disc.update()

                # --- Boson Branch Training (Discriminator update) ---
                for p in boson_discriminator.parameters():
                    p.requires_grad_(True)
                for p in boson_generator.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    boson_latent = get_boson_latent(batch_size).to(device)
                    boson_data_fake = boson_generator(boson_latent).detach()
                with torch.amp.autocast('mps'):
                    boson_real_output = boson_discriminator(data_real)
                    boson_fake_output = boson_discriminator(boson_data_fake)
                    boson_real_target = torch.ones_like(boson_real_output)
                    boson_fake_target = torch.zeros_like(boson_fake_output)
                    boson_disc_loss_real = dice_loss(boson_real_output, boson_real_target)
                    boson_disc_loss_fake = dice_loss(boson_fake_output, boson_fake_target)
                    boson_disc_loss = boson_disc_loss_real + boson_disc_loss_fake
                boson_disc_optimizer.zero_grad()
                boson_scaler_disc.scale(boson_disc_loss).backward()
                boson_disc_optimizer.step()
                boson_scaler_disc.update()

                # --- Boson Nonlinear Branch Training (Discriminator update) ---
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(True)
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    boson_latent_nonlinear = get_boson_latent_nonlinear(batch_size).to(device)
                    boson_data_fake_nonlinear = boson_generator_nonlinear(boson_latent_nonlinear).detach()
                with torch.amp.autocast('mps'):
                    boson_real_output_nonlinear = boson_discriminator_nonlinear(data_real)
                    boson_fake_output_nonlinear = boson_discriminator_nonlinear(boson_data_fake_nonlinear)
                    boson_real_target_nonlinear = torch.ones_like(boson_real_output_nonlinear)
                    boson_fake_target_nonlinear = torch.zeros_like(boson_fake_output_nonlinear)
                    boson_disc_loss_real_nonlinear = dice_loss(boson_real_output_nonlinear, boson_real_target_nonlinear)
                    boson_disc_loss_fake_nonlinear = dice_loss(boson_fake_output_nonlinear, boson_fake_target_nonlinear)
                    boson_disc_loss_nonlinear = boson_disc_loss_real_nonlinear + boson_disc_loss_fake_nonlinear
                boson_disc_optimizer_nonlinear.zero_grad()
                boson_scaler_disc_nonlinear.scale(boson_disc_loss_nonlinear).backward()
                boson_disc_optimizer_nonlinear.step()
                boson_scaler_disc_nonlinear.update()

                # --- PT Branch Training (Discriminator update) ---
                for p in gaussian_discriminator.parameters():
                    p.requires_grad_(True)
                for p in gaussian_generator.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    gaussian_latent = torch.randn(batch_size, latent_dim).to(device)
                    gauss_data_fake = gaussian_generator(gaussian_latent).detach()
                with torch.amp.autocast('mps'):
                    gauss_real_output = gaussian_discriminator(data_real)
                    gauss_fake_output = gaussian_discriminator(gauss_data_fake)
                    gauss_real_target = torch.ones_like(gauss_real_output)
                    gauss_fake_target = torch.zeros_like(gauss_fake_output)
                    gauss_disc_loss_real = dice_loss(gauss_real_output, gauss_real_target)
                    gauss_disc_loss_fake = dice_loss(gauss_fake_output, gauss_fake_target)
                    gauss_disc_loss = gauss_disc_loss_real + gauss_disc_loss_fake
                gaussian_disc_optimizer.zero_grad()
                gauss_scaler_disc.scale(gauss_disc_loss).backward()
                gauss_scaler_disc.step(gaussian_disc_optimizer)
                gauss_scaler_disc.update()

                # --- Generator Training ---
                # PT generator timing start
                pt_gen_start = time.perf_counter()
                # Freeze all discriminators for PT generator update
                for p in pt_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(False)
                for p in gaussian_discriminator.parameters():
                    p.requires_grad_(False)
                # Unfreeze PT generator only
                for p in pt_generator.parameters():
                    p.requires_grad_(True)
                # Ensure other generators remain frozen
                for p in boson_generator.parameters():
                    p.requires_grad_(False)
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(False)
                for p in gaussian_generator.parameters():
                    p.requires_grad_(False)
                # ORCA's simulator
                with torch.amp.autocast('mps'):
                    pt_latent = pt_latent_space.generate(batch_size).to(device)
                    pt_generated_ct = pt_generator(pt_latent)
                    data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                    pt_generated_norm = (pt_generated_ct - pt_generated_ct.min()) / (pt_generated_ct.max() - pt_generated_ct.min())
                    pt_fake_output_for_gen = pt_discriminator(pt_generated_norm)
                    pt_adversarial_loss = dice_loss(pt_fake_output_for_gen, torch.ones_like(pt_fake_output_for_gen))
                    pt_gen_loss = F.l1_loss(pt_generated_norm, data_norm) + 0.5 * pt_adversarial_loss
                pt_gen_optimizer.zero_grad()
                pt_scaler_gen.scale(pt_gen_loss).backward()
                pt_scaler_gen.step(pt_gen_optimizer)
                pt_scaler_gen.update()
                pt_gen_time = time.perf_counter() - pt_gen_start
                print(f"Epoch {epoch+1} PT generator training time: {pt_gen_time:.3f} seconds")

                # Boson generator timing start
                boson_gen_start = time.perf_counter()
                # Freeze all discriminators for Boson generator update
                for p in pt_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(False)
                for p in gaussian_discriminator.parameters():
                    p.requires_grad_(False)
                # Unfreeze Boson generator only
                for p in boson_generator.parameters():
                    p.requires_grad_(True)
                # Ensure other generators remain frozen
                for p in pt_generator.parameters():
                    p.requires_grad_(False)
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(False)
                for p in gaussian_generator.parameters():
                    p.requires_grad_(False)
                # Our simulator
                with torch.amp.autocast('mps'):
                    boson_latent = get_boson_latent(batch_size).to(device)
                    boson_generated_ct = boson_generator(boson_latent)
                    data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                    boson_generated_norm = (boson_generated_ct - boson_generated_ct.min()) / (boson_generated_ct.max() - boson_generated_ct.min())
                    boson_fake_output_for_gen = boson_discriminator(boson_generated_norm)
                    boson_adversarial_loss = dice_loss(boson_fake_output_for_gen, torch.ones_like(boson_fake_output_for_gen))
                    boson_gen_loss = F.l1_loss(boson_generated_norm, data_norm) + 0.5 * boson_adversarial_loss
                boson_gen_optimizer.zero_grad()
                boson_scaler_gen.scale(boson_gen_loss).backward()
                boson_gen_optimizer.step()
                boson_scaler_gen.update()
                boson_gen_time = time.perf_counter() - boson_gen_start
                print(f"Epoch {epoch+1} Boson generator training time: {boson_gen_time:.3f} seconds")

                # Boson Nonlinear generator timing start
                boson_nl_gen_start = time.perf_counter()
                # Freeze all discriminators for Boson Nonlinear generator update
                for p in pt_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(False)
                for p in gaussian_discriminator.parameters():
                    p.requires_grad_(False)
                # Unfreeze Boson Nonlinear generator only
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(True)
                # Ensure other generators remain frozen
                for p in pt_generator.parameters():
                    p.requires_grad_(False)
                for p in boson_generator.parameters():
                    p.requires_grad_(False)
                for p in gaussian_generator.parameters():
                    p.requires_grad_(False)
                # Boson Nonlinear Generator
                with torch.amp.autocast('mps'):
                    boson_latent_nonlinear = get_boson_latent_nonlinear(batch_size).to(device)
                    boson_generated_ct_nonlinear = boson_generator_nonlinear(boson_latent_nonlinear)
                    data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                    boson_generated_norm_nonlinear = (boson_generated_ct_nonlinear - boson_generated_ct_nonlinear.min()) / (boson_generated_ct_nonlinear.max() - boson_generated_ct_nonlinear.min())
                    boson_fake_output_for_gen_nonlinear = boson_discriminator_nonlinear(boson_generated_norm_nonlinear)
                    boson_adversarial_loss_nonlinear = dice_loss(boson_fake_output_for_gen_nonlinear, torch.ones_like(boson_fake_output_for_gen_nonlinear))
                    boson_gen_loss_nonlinear = F.l1_loss(boson_generated_norm_nonlinear, data_norm) + 0.5 * boson_adversarial_loss_nonlinear
                boson_gen_optimizer_nonlinear.zero_grad()
                boson_scaler_gen_nonlinear.scale(boson_gen_loss_nonlinear).backward()
                boson_gen_optimizer_nonlinear.step()
                boson_scaler_gen_nonlinear.update()
                boson_nl_gen_time = time.perf_counter() - boson_nl_gen_start
                print(f"Epoch {epoch+1} Boson NL generator training time: {boson_nl_gen_time:.3f} seconds")

                # Gaussian generator timing start
                gaussian_gen_start = time.perf_counter()
                # Freeze all discriminators for Gaussian generator update
                for p in pt_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator.parameters():
                    p.requires_grad_(False)
                for p in boson_discriminator_nonlinear.parameters():
                    p.requires_grad_(False)
                for p in gaussian_discriminator.parameters():
                    p.requires_grad_(False)
                # Unfreeze Gaussian generator only
                for p in gaussian_generator.parameters():
                    p.requires_grad_(True)
                # Ensure other generators remain frozen
                for p in pt_generator.parameters():
                    p.requires_grad_(False)
                for p in boson_generator.parameters():
                    p.requires_grad_(False)
                for p in boson_generator_nonlinear.parameters():
                    p.requires_grad_(False)
                # Gaussian
                with torch.amp.autocast('mps'):
                    gauss_latent = torch.randn(batch_size, latent_dim).to(device)
                    gauss_generated_ct = gaussian_generator(gauss_latent)
                    data_norm = (data_real - data_real.min()) / (data_real.max() - data_real.min())
                    gauss_generated_norm = (gauss_generated_ct - gauss_generated_ct.min()) / (gauss_generated_ct.max() - gauss_generated_ct.min())
                    gauss_fake_output_for_gen = gaussian_discriminator(gauss_generated_norm)
                    gauss_adversarial_loss = dice_loss(gauss_fake_output_for_gen, torch.ones_like(gauss_fake_output_for_gen))
                    gauss_gen_loss = F.l1_loss(gauss_generated_norm, data_norm) + 0.5 * gauss_adversarial_loss
                gaussian_gen_optimizer.zero_grad()
                gauss_scaler_gen.scale(gauss_gen_loss).backward()
                gauss_scaler_gen.step(gaussian_gen_optimizer)
                gauss_scaler_gen.update()
                gaussian_gen_time = time.perf_counter() - gaussian_gen_start
                print(f"Epoch {epoch+1} Gaussian generator training time: {gaussian_gen_time:.3f} seconds")

                log_csv(run+1, seed, "Main Training", epoch+1, pt_gen_loss.item(), boson_gen_loss.item(), boson_gen_loss_nonlinear.item(), gauss_gen_loss.item(), csv_file)

                print(f"Epoch {epoch+1}, PT Gen Loss: {pt_gen_loss.item()}, PT Disc Loss: {pt_disc_loss.item()}, " +
                    f"Boson Gen Loss: {boson_gen_loss.item()}, Boson Disc Loss: {boson_disc_loss.item()}" + 
                    f"Boson Gen Nonlinear Loss: {boson_gen_loss_nonlinear.item()}, Boson Disc Loss: {boson_disc_loss_nonlinear.item()}" + 
                    f"Gaussian Gen Loss: {gauss_gen_loss.item()}, Boson Disc Loss: {gauss_disc_loss.item()}")

                if (epoch + 1) % 250 == 0:
                    model_save_folder = 'model_checkpoints'
                    os.makedirs(model_save_folder, exist_ok=True)
                    pt_generator_save_path = os.path.join(model_save_folder, f'pt_generator_epoch_{epoch + 1}.pt')
                    pt_discriminator_save_path = os.path.join(model_save_folder, f'pt_discriminator_epoch_{epoch + 1}.pt')
                    boson_generator_save_path = os.path.join(model_save_folder, f'boson_generator_epoch_{epoch + 1}.pt')
                    boson_discriminator_save_path = os.path.join(model_save_folder, f'boson_discriminator_epoch_{epoch + 1}.pt')
                    boson_generator_nonlinear_save_path = os.path.join(model_save_folder, f'boson_nonlinear_generator_epoch_{epoch + 1}.pt')
                    boson_discriminator_nonlinear_save_path = os.path.join(model_save_folder, f'boson_nonlinear_discriminator_epoch_{epoch + 1}.pt')
                    gaussian_generator_nonlinear_save_path = os.path.join(model_save_folder, f'gaussian_generator_epoch_{epoch + 1}.pt')
                    gaussian_discriminator_nonlinear_save_path = os.path.join(model_save_folder, f'gaussian_discriminator_epoch_{epoch + 1}.pt')
                    
                    save_weights(pt_generator, pt_gen_optimizer, epoch + 1, pt_generator_save_path, pt_gen_loss.item())
                    save_weights(pt_discriminator, pt_disc_optimizer, epoch + 1, pt_discriminator_save_path, pt_disc_loss.item())
                    save_weights(boson_generator, boson_gen_optimizer, epoch + 1, boson_generator_save_path, boson_gen_loss.item())
                    save_weights(boson_discriminator, boson_disc_optimizer, epoch + 1, boson_discriminator_save_path, boson_disc_loss.item())
                    save_weights(boson_generator, boson_gen_optimizer_nonlinear, epoch + 1, boson_generator_nonlinear_save_path, boson_gen_loss_nonlinear.item())
                    save_weights(boson_discriminator, boson_disc_optimizer_nonlinear, epoch + 1, boson_discriminator_nonlinear_save_path, boson_disc_loss_nonlinear.item())
                    save_weights(gaussian_generator, gaussian_gen_optimizer, epoch + 1, gaussian_generator_nonlinear_save_path, gauss_gen_loss.item())
                    save_weights(gaussian_discriminator, gaussian_disc_optimizer, epoch + 1, gaussian_discriminator_nonlinear_save_path, gauss_disc_loss.item())
                if (epoch + 1) % 50 == 0:
                    with torch.no_grad():
                        # Use fixed latents for evaluation to reduce variance
                        pt_generated = pt_generator(fixed_pt_latent)
                        boson_generated = boson_generator(fixed_boson_latent)
                        boson_generated_nl = boson_generator_nonlinear(fixed_boson_nl_latent)
                        gauss_generated = gaussian_generator(fixed_gauss_latent)
                        # Compute FID scores using the same real images batch
                        fid_pt = compute_fid(data_real, pt_generated, inception_model, device)
                        fid_boson = compute_fid(data_real, boson_generated, inception_model, device)
                        fid_boson_nl = compute_fid(data_real, boson_generated_nl, inception_model, device)
                        fid_gauss = compute_fid(data_real, gauss_generated, inception_model, device)
                    fid_pt_list.append(fid_pt)
                    fid_boson_list.append(fid_boson)
                    fid_boson_nl_list.append(fid_boson_nl)
                    fid_gauss_list.append(fid_gauss)
                    epoch_list.append(epoch+1)
                    print(f"Epoch {epoch+1} FID - PT: {fid_pt:.3f}, Boson: {fid_boson:.3f}, Boson NL: {fid_boson_nl:.3f}, Gaussian: {fid_gauss:.3f}")
                pt_gen_scheduler.step(pt_gen_loss)
                pt_disc_scheduler.step(pt_disc_loss)
                boson_gen_scheduler.step(boson_gen_loss)
                boson_disc_scheduler.step(boson_disc_loss)
                boson_gen_nonlinear_scheduler.step(boson_gen_loss_nonlinear)
                boson_disc_nonlinear_scheduler.step(boson_disc_loss_nonlinear)
                gauss_gen_scheduler.step(gauss_gen_loss)
                gauss_disc_scheduler.step(gauss_disc_loss)
            print(f'Run {run+1}, PT Gen Loss: {pt_gen_loss.item()}, PT Disc Loss: {pt_disc_loss.item()}, ' +
                f'Boson Gen Loss: {boson_gen_loss.item()}, Boson Disc Loss: {boson_disc_loss.item()}' +
                f'Boson Gen Loss: {boson_gen_loss_nonlinear.item()}, Boson Disc Loss: {boson_disc_loss_nonlinear.item()}' +
                f'Boson Gen Loss: {gauss_gen_loss.item()}, Boson Disc Loss: {gauss_disc_loss.item()}')
            # Store this seed's FID curve
            avg_results['pt'].append(fid_pt_list)
            avg_results['boson'].append(fid_boson_list)
            avg_results['boson_nl'].append(fid_boson_nl_list)
            avg_results['gauss'].append(fid_gauss_list)
    # Compute mean FID across all seeds for each epoch
    mean_fid_pt = np.mean(avg_results['pt'], axis=0)
    mean_fid_boson = np.mean(avg_results['boson'], axis=0)
    mean_fid_boson_nl = np.mean(avg_results['boson_nl'], axis=0)
    mean_fid_gauss = np.mean(avg_results['gauss'], axis=0)
    # Plot mean FID curves
    plt.figure()
    plt.plot(epoch_list, mean_fid_pt, label='PT Mean FID')
    plt.plot(epoch_list, mean_fid_boson, label='Boson Mean FID')
    plt.plot(epoch_list, mean_fid_boson_nl, label='Boson NL Mean FID')
    plt.plot(epoch_list, mean_fid_gauss, label='Gaussian Mean FID')
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
