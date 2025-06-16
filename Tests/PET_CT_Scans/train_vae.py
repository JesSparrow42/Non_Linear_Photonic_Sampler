import os
import csv
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Optional
import random

# Example imports for data loading—replace with your own
from .utils.data_loader import create_data_loader
# If your data loader returns CT images, be sure it returns them in
# a shape that matches your VAE's input_shape (e.g. (1,128,128))

# Import your VAE modules
# (Either put VariationalAutoencoder and VariationalInference in this file or import them.)
from .vae_module import VariationalAutoencoder, VariationalInference

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic   = True 
torch.backends.cudnn.benchmark       = False
# ------------------------------------------------------------------------
# Example training script for a VAE, similar in style to your train.py
# ------------------------------------------------------------------------
def train_vae(
    data_loader: DataLoader,
    input_shape: tuple,
    latent_dim: int = 32,
    lr: float = 1e-4,
    beta: float = 1.0,
    n_epochs: int = 1000,
    csv_filename: str = 'vae_training_log.csv',
    device: Optional[torch.device] = None,
    boson_sampler_params: Optional[dict] = None
):
    """
    data_loader: A PyTorch DataLoader yielding batches of images shaped (B, C, H, W).
    input_shape: The shape of a single image, e.g. (1, 128, 128) for grayscale 128x128.
    latent_dim:  Number of latent features in the VAE.
    lr:          Learning rate for the optimizer.
    beta:        Beta for beta-VAE if desired; set to 1.0 for standard VAE.
    n_epochs:    Number of training epochs.
    csv_filename: Path to CSV file for logging.
    device:      Which device to use; if None, chooses CUDA or CPU automatically.
    """

    # ---------------------------
    # Prepare device
    # ---------------------------
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            print("Using CPU")

    # ---------------------------
    # Initialize VAE
    # ---------------------------
    # NOTE: boson_sampler_params=None => we only use a standard Gaussian prior
    vae = VariationalAutoencoder(
        input_shape=input_shape,
        latent_features=latent_dim,
        boson_sampler_params=boson_sampler_params
    ).to(device)

    # The VAE objective module
    elbo_function = VariationalInference(beta=beta)

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    # ---------------------------
    # Setup CSV logging
    # ---------------------------
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "AverageLoss", "AverageLogPx", "AverageKL"])

    # ---------------------------
    # Training Loop
    # ---------------------------
    vae.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_log_px = 0.0
        total_kl = 0.0
        count_batches = 0

        for batch_data in data_loader:
            # If loader returns (CT, PET), just take CT images (first element)
            if isinstance(batch_data, (list, tuple)):
                batch_data = batch_data[0]
            x = batch_data.to(device).float()
            # scale each volume to [0,1]
            x = (x - x.min()) / (x.max() - x.min() + 1e-6)
            # optionally shift to [–1,+1]
            x = x * 2 - 1
            optimizer.zero_grad()

            # Forward pass / compute loss
            loss, diagnostics, _ = elbo_function(vae, x)
            loss.backward()
            optimizer.step()

            # Stats
            total_loss += loss.item()
            total_log_px += diagnostics['log_px'].mean().item()   # average over batch
            total_kl += diagnostics['kl'].mean().item()
            count_batches += 1

        # Averages per epoch
        avg_loss = total_loss / count_batches
        avg_log_px = total_log_px / count_batches
        avg_kl = total_kl / count_batches

        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, log_px: {avg_log_px:.4f}, KL: {avg_kl:.4f}")

        # Write to CSV
        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_loss, avg_log_px, avg_kl])

    print("Training complete! Model is trained.")

    # Return the trained VAE in case you want to do further analysis
    return vae


def main():
    # ---------------------------
    # Example config/hyperparams
    # ---------------------------
    ct_folder = "/Users/olivernorregaard/Downloads/DiscretePETCT-main/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805"
    pet_folder = "/Users/olivernorregaard/Downloads/DiscretePETCT-main/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/1.000000-PET NAC-24000"
    batch_size = 4
    num_workers = 2
    n_epochs = 100
    latent_dim = 32
    lr = 1e-4
    beta = 1.0  # Standard VAE

    # Example: get a data loader that yields images in shape (B, 1, 128, 128)
    data_loader = create_data_loader(
        ct_folder=ct_folder,
        pet_folder=pet_folder,
        num_workers=num_workers,
        augment=False,
        batch_size=batch_size,
        shuffle=False
    )

    # The shape that the VAE expects for each single image
    input_shape = (1, 128, 128)  # or adapt based on your dataset

    # --- First run: Gaussian prior (boson_sampler_params=None)
    print("Running VAE with Gaussian prior...")
    trained_vae_gaussian = train_vae(
        data_loader=data_loader,
        input_shape=input_shape,
        latent_dim=latent_dim,
        lr=lr,
        beta=beta,
        n_epochs=n_epochs,
        csv_filename='vae_training_log_gaussian.csv',
        device=None,  # auto-selects device
        boson_sampler_params=None
    )

    torch.save(trained_vae_gaussian.state_dict(), "trained_vae_gaussian.pth")

    # --- Second run: PTGenerator prior (pass a dictionary of PTGenerator params)
    # Adjust these param names/values as appropriate for your PTGenerator
    ptgen_params = {
        "input_state": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "tbi_params": {
            "input_loss": 0.0,
            "detector_efficiency": 1,
            "bs_loss": 0,
            "bs_noise": 0,
            "distinguishable": False,
            "n_signal_detectors": 0,
            "g2": 0,
            "tbi_type": "multi-loop",
            "n_loops": 2,
            "loop_lengths": [1, 2],
            "postselected": True
        },
        "n_tiling": 1
    }

    print("Running VAE with PTGenerator prior...")
    trained_vae_ptgen = train_vae(
        data_loader=data_loader,
        input_shape=input_shape,
        latent_dim=latent_dim,
        lr=lr,
        beta=beta,
        n_epochs=n_epochs,
        csv_filename='vae_training_log_ptgen.csv',
        device=None,
        boson_sampler_params=ptgen_params
    )

    torch.save(trained_vae_ptgen.state_dict(), "trained_vae_ptgen.pth")

    # --- Third run: Boson Sampler linear prior
    bs_linear_params = {
        "m": latent_dim,
        "num_sources": latent_dim // 2,
        "num_loops": 100,
        "input_loss": 0.0,
        "coupling_efficiency": 1.0,
        "detector_inefficiency": 1.0,
        "mu": 1.0,
        "temporal_mismatch": 0.0,
        "spectral_mismatch": 0.0,
        "arrival_time_jitter": 0.0,
        "bs_loss": 0.0,
        "bs_jitter": 0.0,
        "phase_noise_std": 0.0,
        "systematic_phase_offset": 0.0,
        "mode_loss": [1.0] * latent_dim,
        "dark_count_rate": 0.0,
        "use_advanced_nonlinearity": False
    }
    print("Running VAE with Boson linear prior...")
    trained_vae_boson_linear = train_vae(
        data_loader=data_loader,
        input_shape=input_shape,
        latent_dim=latent_dim,
        lr=lr,
        beta=beta,
        n_epochs=n_epochs,
        csv_filename='vae_training_log_boson_linear.csv',
        device=None,
        boson_sampler_params=bs_linear_params
    )
    torch.save(trained_vae_boson_linear.state_dict(), "trained_vae_boson_linear.pth")

    # --- Fourth run: Boson Sampler nonlinear prior
    bs_nl_params = dict(bs_linear_params)
    bs_nl_params["use_advanced_nonlinearity"] = True
    print("Running VAE with Boson non-linear prior...")
    trained_vae_boson_nl = train_vae(
        data_loader=data_loader,
        input_shape=input_shape,
        latent_dim=latent_dim,
        lr=lr,
        beta=beta,
        n_epochs=n_epochs,
        csv_filename='vae_training_log_boson_nonlinear.csv',
        device=None,
        boson_sampler_params=bs_nl_params
    )
    torch.save(trained_vae_boson_nl.state_dict(), "trained_vae_boson_nonlinear.pth")


if __name__ == "__main__":
    main()