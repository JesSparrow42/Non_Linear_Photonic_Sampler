import os
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from mnist_gan import Generator, Discriminator
from bosonsampler import BosonLatentGenerator, BosonSamplerTorch
from ptseries.models import PTGenerator

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ---------------------------
# Gradient Penalty for WGAN-GP
# ---------------------------
def gradient_penalty(critic, real, fake, device):
    """
    Computes gradient penalty for WGAN-GP.
    :param critic: The discriminator/critic network.
    :param real: Real images batch (B, C, H, W).
    :param fake: Fake images batch (B, C, H, W).
    :param device: torch device (cpu or cuda).
    :return: scalar gradient penalty term.
    """
    batch_size = real.size(0)
    # Random weight for interpolation between real and fake
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
    
    # Interpolate between real and fake
    interpolated_images = epsilon * real + (1 - epsilon) * fake
    interpolated_images = interpolated_images.to(device)
    # Forward pass through critic
    mixed_scores = critic(interpolated_images)
    # Compute gradients wrt the interpolated images
    grad_outputs = torch.ones_like(mixed_scores, device=device)
    gradients = torch.autograd.grad(
        outputs=mixed_scores,
        inputs=interpolated_images,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    # Flatten the gradients to compute norm
    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

# ---------------------------
# Main training script (WGAN-GP)
# ---------------------------
def main():
    # Hyperparameters
    NUM_EPOCHS = 50
    BATCH_SIZE = 128
    CRITIC_LR = 1e-4
    GEN_LR = 1e-4
    LATENT_DIM = 10  # must match the dimension used by your PTGenerator and BosonSampler
    LAMBDA_GP = 10.0  # Gradient penalty lambda
    N_CRITIC = 3      # Number of critic iterations per generator iteration (often 5 for WGAN-GP)

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    print("Using device:", device)

    # ---------------------------
    # Create latent generators for the three branches
    # ---------------------------
    boson_sampler_params = {
        "input_state": [1, 0, 1,0,1,0,1,0,1,0],
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

    pt_latent_space = PTGenerator(**boson_sampler_params)  # PT branch latent
    boson_sampler_torch = BosonSamplerTorch(
        m=LATENT_DIM,
        num_sources=LATENT_DIM // 2,
        num_loops=50,
        input_loss=1,
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
        mode_loss=np.ones(LATENT_DIM),
        dark_count_rate=0,
        use_advanced_nonlinearity=False,
        g2_target=1,
    )
    boson_latent_space = BosonLatentGenerator(LATENT_DIM, boson_sampler_torch)

    boson_sampler_torch_nl = BosonSamplerTorch(
        m=LATENT_DIM,
        num_sources=LATENT_DIM // 2,
        num_loops=50,
        input_loss=0,
        coupling_efficiency=1,
        detector_inefficiency=1,
        g2_target=1,
        mu=1,
        temporal_mismatch=0,
        spectral_mismatch=0,
        arrival_time_jitter=0,
        bs_loss=1,
        bs_jitter=0,
        phase_noise_std=0,
        systematic_phase_offset=0,
        mode_loss=np.ones(LATENT_DIM),
        dark_count_rate=0,
        use_advanced_nonlinearity=True,
        detuning=0.0,
        pulse_bw=0.0,
        QD_linewidth=1.0,
        phi=0.0,
    )
    boson_latent_space_nl = BosonLatentGenerator(LATENT_DIM, boson_sampler_torch_nl)

    # ---------------------------
    # Create Generators and Critics for each branch
    # ---------------------------
    pt_generator = Generator(LATENT_DIM).to(device)
    pt_critic = Discriminator().to(device)  # rename "discriminator" -> "critic" for clarity

    boson_generator = Generator(LATENT_DIM).to(device)
    boson_critic = Discriminator().to(device)

    boson_generator_nl = Generator(LATENT_DIM).to(device)
    boson_critic_nl = Discriminator().to(device)

    # ---------------------------
    # Set up Optimizers
    # ---------------------------
    pt_gen_optimizer = optim.Adam(pt_generator.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
    pt_critic_optimizer = optim.Adam(pt_critic.parameters(), lr=CRITIC_LR, betas=(0.5, 0.999))

    boson_gen_optimizer = optim.Adam(boson_generator.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
    boson_critic_optimizer = optim.Adam(boson_critic.parameters(), lr=CRITIC_LR, betas=(0.5, 0.999))

    boson_gen_nl_optimizer = optim.Adam(boson_generator_nl.parameters(), lr=GEN_LR, betas=(0.5, 0.999))
    boson_critic_nl_optimizer = optim.Adam(boson_critic_nl.parameters(), lr=CRITIC_LR, betas=(0.5, 0.999))

    # ---------------------------
    # Load MNIST dataset
    # ---------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST in [-1, 1]
    ])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    subset_size = 10000
    subset_indices = list(range(subset_size))
    mnist_subset = Subset(full_dataset, subset_indices)
    data_loader = DataLoader(
        mnist_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
    )

    # CSV logging setup
    csv_file = 'wgan_gp_mnist.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Run', 'Phase', 'Epoch',
                'PT_Gen_Loss', 'PT_Critic_Loss',
                'Boson_Gen_Loss', 'Boson_Critic_Loss',
                'BosonNL_Gen_Loss', 'BosonNL_Critic_Loss',
                'Timestamp'
            ])

    # ---------------------------
    # Training Loop
    # ---------------------------
    for epoch in range(1, NUM_EPOCHS + 1):
        for batch_idx, (real_images, _) in enumerate(data_loader):
            real_images = real_images.to(device, non_blocking=True)

            # Pre-sample latents for critic updates
            pt_latents = pt_latent_space.generate(N_CRITIC * BATCH_SIZE).to(device)
            pt_latents = pt_latents.chunk(N_CRITIC)
            boson_latents = boson_latent_space(BATCH_SIZE * N_CRITIC).to(device)
            boson_latents = boson_latents.chunk(N_CRITIC)
            boson_nl_latents = boson_latent_space_nl(BATCH_SIZE * N_CRITIC).to(device)
            boson_nl_latents = boson_nl_latents.chunk(N_CRITIC)

            # ---------------------------
            # TRAIN CRITICS (multiple iterations)
            # ---------------------------
            for i in range(N_CRITIC):
                # ---- PT BRANCH ----
                pt_critic_optimizer.zero_grad()
                pt_latent = pt_latents[i]
                pt_fake = pt_generator(pt_latent)

                # Critic outputs
                pt_real_out = pt_critic(real_images)
                pt_fake_out = pt_critic(pt_fake)

                # Gradient penalty
                gp_pt = gradient_penalty(pt_critic, real_images, pt_fake, device)

                # WGAN-GP critic loss
                pt_critic_loss = -(pt_real_out.mean() - pt_fake_out.mean()) + LAMBDA_GP * gp_pt
                pt_critic_loss.backward()
                pt_critic_optimizer.step()

                # ---- BOSON BRANCH ----
                boson_critic_optimizer.zero_grad()
                boson_latent = boson_latents[i]
                boson_fake = boson_generator(boson_latent)

                boson_real_out = boson_critic(real_images)
                boson_fake_out = boson_critic(boson_fake)
                gp_boson = gradient_penalty(boson_critic, real_images, boson_fake, device)
                boson_critic_loss = -(boson_real_out.mean() - boson_fake_out.mean()) + LAMBDA_GP * gp_boson
                boson_critic_loss.backward()
                boson_critic_optimizer.step()

                # ---- BOSON NONLINEAR BRANCH ----
                boson_critic_nl_optimizer.zero_grad()
                boson_nl_latent = boson_nl_latents[i]
                boson_nl_fake = boson_generator_nl(boson_nl_latent)

                boson_nl_real_out = boson_critic_nl(real_images)
                boson_nl_fake_out = boson_critic_nl(boson_nl_fake)
                gp_boson_nl = gradient_penalty(boson_critic_nl, real_images, boson_nl_fake, device)
                boson_nl_critic_loss = -(boson_nl_real_out.mean() - boson_nl_fake_out.mean()) + LAMBDA_GP * gp_boson_nl
                boson_nl_critic_loss.backward()
                boson_critic_nl_optimizer.step()

            # ---------------------------
            # TRAIN GENERATORS (1 iteration after N_CRITIC)
            # ---------------------------
            # ---- PT BRANCH GEN ----
            pt_gen_optimizer.zero_grad()
            pt_latent = pt_latent_space.generate(BATCH_SIZE).to(device)
            pt_fake = pt_generator(pt_latent)
            pt_fake_out = pt_critic(pt_fake)
            pt_gen_loss = -pt_fake_out.mean()  # WGAN-GP generator loss
            pt_gen_loss.backward()
            pt_gen_optimizer.step()

            # ---- BOSON BRANCH GEN ----
            boson_gen_optimizer.zero_grad()
            boson_latent = boson_latent_space(BATCH_SIZE).to(device)
            boson_fake = boson_generator(boson_latent)
            boson_fake_out = boson_critic(boson_fake)
            boson_gen_loss = -boson_fake_out.mean()
            boson_gen_loss.backward()
            boson_gen_optimizer.step()

            # ---- BOSON NONLINEAR BRANCH GEN ----
            boson_gen_nl_optimizer.zero_grad()
            boson_nl_latent = boson_latent_space_nl(BATCH_SIZE).to(device)
            boson_nl_fake = boson_generator_nl(boson_nl_latent)
            boson_nl_fake_out = boson_critic_nl(boson_nl_fake)
            boson_nl_gen_loss = -boson_nl_fake_out.mean()
            boson_nl_gen_loss.backward()
            boson_gen_nl_optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch [{batch_idx}/{len(data_loader)}] | "
                    f"PT Critic: {pt_critic_loss.item():.4f}, PT Gen: {pt_gen_loss.item():.4f} | "
                    f"Boson Critic: {boson_critic_loss.item():.4f}, Boson Gen: {boson_gen_loss.item():.4f} | "
                    f"BosonNL Critic: {boson_nl_critic_loss.item():.4f}, BosonNL Gen: {boson_nl_gen_loss.item():.4f}"
                )

        # ---------------------------
        # End of Epoch: Logging, Checkpoint, etc.
        # ---------------------------
        # Log to CSV (just record the final batch losses from this epoch)
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "WGAN-GP", "Epoch", epoch,
                pt_gen_loss.item(), pt_critic_loss.item(),
                boson_gen_loss.item(), boson_critic_loss.item(),
                boson_nl_gen_loss.item(), boson_nl_critic_loss.item(),
                datetime.now()
            ])

        # Save model checkpoints every 10 epochs
        if epoch % 10 == 0:
            os.makedirs('model_checkpoints', exist_ok=True)
            torch.save(pt_generator.state_dict(), f'model_checkpoints/pt_generator_epoch_{epoch}.pt')
            torch.save(pt_critic.state_dict(), f'model_checkpoints/pt_critic_epoch_{epoch}.pt')
            torch.save(boson_generator.state_dict(), f'model_checkpoints/boson_generator_epoch_{epoch}.pt')
            torch.save(boson_critic.state_dict(), f'model_checkpoints/boson_critic_epoch_{epoch}.pt')
            torch.save(boson_generator_nl.state_dict(), f'model_checkpoints/boson_generator_nl_epoch_{epoch}.pt')
            torch.save(boson_critic_nl.state_dict(), f'model_checkpoints/boson_critic_nl_epoch_{epoch}.pt')

    print("WGAN-GP training complete.")

if __name__ == "__main__":
    main()
