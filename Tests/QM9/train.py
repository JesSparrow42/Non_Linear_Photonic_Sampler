import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import time

# Silence RDKit console spam (warnings + valence errors)
from rdkit import RDLogger
for level in ('rdApp.debug', 'rdApp.info', 'rdApp.warning', 'rdApp.error'):
    RDLogger.DisableLog(level)

import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

# Use deterministic algorithms to mitigate NaN/Inf issues
torch.use_deterministic_algorithms(True, warn_only=True)

import numpy as np
import random
import csv
import os
import wandb
from bosonsampler import BosonLatentGenerator, BosonSamplerTorch

from dataset import SparseMolecularDataset
from models import MolGANGenerator, MolGANDiscriminator, MolGANReward
from utils import compute_gradient_penalty, sample_z
from molecular_metrics import MolecularMetrics


# -- Sampler timing helper --
def measure_sampler_time(fn, *, n_latents=128, repeats=20, device_str="cpu"):
    """
    Return average milliseconds per latent for `fƒn(batch_size)`.
    """
    # warm up
    _ = fn(8)
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = fn(n_latents)
        if device_str == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    return ms / (repeats * n_latents)

def train(args, seed=None, writer=None):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    num_workers = 0 if device.type == 'mps' else 4

    # Enable cuDNN autotuner for faster convolutions
    if device.type == 'cuda':
        cudnn.benchmark = True

    # -- data --
    ds = SparseMolecularDataset(args.data_path, subset=args.subset)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=(device.type=='cuda'),
    )

    # -- models for three branches --
    G_base = MolGANGenerator(args.z_dim_base, ds.node_features.shape[1],
                             ds.node_features.shape[2], ds.adjacency.shape[3],
                             hidden_dim=args.hidden_dim_base, tau=args.tau_base).to(device)
    D_base = MolGANDiscriminator(ds.node_features.shape[2],
                                 ds.adjacency.shape[3],
                                 hidden_dim=args.hidden_dim_base).to(device)
    R_base = MolGANReward(ds.node_features.shape[2],
                          ds.adjacency.shape[3],
                          hidden_dim=args.hidden_dim_base).to(device)
    G_bos = MolGANGenerator(args.z_dim_bos, ds.node_features.shape[1],
                            ds.node_features.shape[2], ds.adjacency.shape[3],
                            hidden_dim=args.hidden_dim_bos, tau=args.tau_bos).to(device)
    D_bos = MolGANDiscriminator(ds.node_features.shape[2],
                                ds.adjacency.shape[3],
                                hidden_dim=args.hidden_dim_bos).to(device)
    R_bos = MolGANReward(ds.node_features.shape[2],
                         ds.adjacency.shape[3],
                         hidden_dim=args.hidden_dim_bos).to(device)
    G_nl = MolGANGenerator(args.z_dim_nl, ds.node_features.shape[1],
                           ds.node_features.shape[2], ds.adjacency.shape[3],
                           hidden_dim=args.hidden_dim_nl, tau=args.tau_nl).to(device)
    D_nl = MolGANDiscriminator(ds.node_features.shape[2],
                               ds.adjacency.shape[3],
                               hidden_dim=args.hidden_dim_nl).to(device)
    R_nl = MolGANReward(ds.node_features.shape[2],
                        ds.adjacency.shape[3],
                        hidden_dim=args.hidden_dim_nl).to(device)

    # Torch‑compile speeds up CPU/CUDA but is unstable on MPS/Metal.
    def maybe_compile(model):
        if hasattr(torch, "compile") and device.type != "mps":
            # safe fallback: disable inductor on unsupported back‑ends
            return torch.compile(model, backend="inductor")
        return model

    G_base = maybe_compile(G_base)
    D_base = maybe_compile(D_base)
    R_base = maybe_compile(R_base)
    G_bos  = maybe_compile(G_bos)
    D_bos  = maybe_compile(D_bos)
    R_bos  = maybe_compile(R_bos)
    G_nl   = maybe_compile(G_nl)
    D_nl   = maybe_compile(D_nl)
    R_nl   = maybe_compile(R_nl)

    # -- optimizers for each branch --
    optG_base = Adam(G_base.parameters(), lr=args.lr_base, betas=(0.9, 0.999))
    optD_base = Adam(D_base.parameters(), lr=args.lr_base, betas=(0.9, 0.999))
    optR_base = Adam(R_base.parameters(), lr=args.lr_base, betas=(0.9, 0.999))
    optG_bos = Adam(G_bos.parameters(), lr=args.lr_bos, betas=(0.9, 0.999))
    optD_bos = Adam(D_bos.parameters(), lr=args.lr_bos, betas=(0.9, 0.999))
    optR_bos = Adam(R_bos.parameters(), lr=args.lr_bos, betas=(0.9, 0.999))
    optG_nl = Adam(G_nl.parameters(), lr=args.lr_nl, betas=(0.9, 0.999))
    optD_nl = Adam(D_nl.parameters(), lr=args.lr_nl, betas=(0.9, 0.999))
    optR_nl = Adam(R_nl.parameters(), lr=args.lr_nl, betas=(0.9, 0.999))

    metrics = MolecularMetrics(weights={
        'validity':args.w_valid, 'qed':args.w_qed,
        'logp':args.w_logp, 'sa':args.w_sa
    })

    # -- setup latent spaces --
    # Baseline Gaussian sampler
    def baseline_latent(bs): return sample_z(bs, args.z_dim_base, device)
    # Prepare boson-based latent spaces
    input_state_bos = [1 if i % 2 == 0 else 0 for i in range(args.z_dim_bos)]
    input_state_nl = [1 if i % 2 == 0 else 0 for i in range(args.z_dim_nl)]
    bs_torch = BosonSamplerTorch(
        early_late_pairs=args.z_dim_bos,
        input_state=input_state_bos,
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
        mode_loss=np.ones(args.z_dim_bos),
        dark_count_rate=0,
        use_advanced_nonlinearity=False,
        g2_target=0
    )
    boson_latent_space = BosonLatentGenerator(args.z_dim_bos, bs_torch)
    # Nonlinear boson sampler
    bs_torch_nl = BosonSamplerTorch(
        early_late_pairs=args.z_dim_nl,
        input_state=input_state_nl,
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
        mode_loss=np.ones(args.z_dim_nl),
        dark_count_rate=0,
        use_advanced_nonlinearity=True,
        pulse_bw=0.5,
        detuning=0.0,
        phi=0.0,
        g2_target=0
    )
    boson_nl_latent_space = BosonLatentGenerator(args.z_dim_nl, bs_torch_nl)
    # Sampler functions for each branch
    sampler_fn_base = baseline_latent
    sampler_fn_bos = boson_latent_space
    sampler_fn_nl = boson_nl_latent_space

    # -- latent space sampler benchmark for all variants --
    if args.benchmark:
        baseline_lat_ms = measure_sampler_time(sampler_fn_base, n_latents=args.batch_size, repeats=20, device_str=device.type)
        boson_lat_ms   = measure_sampler_time(sampler_fn_bos, n_latents=args.batch_size, repeats=20, device_str=device.type)
        boson_nl_lat_ms = measure_sampler_time(sampler_fn_nl, n_latents=args.batch_size, repeats=20, device_str=device.type)
        print(
            f"[Sampler latency] Gaussian: {baseline_lat_ms:.4f} ms | "
            f"Boson: {boson_lat_ms:.4f} ms | "
            f"Boson-NL: {boson_nl_lat_ms:.4f} ms"
        )
    # -- fixed latent for evaluation for all variants --
    with torch.no_grad():
        fixed_baseline_z = sampler_fn_base(args.batch_size).to(device)
        fixed_boson_z    = sampler_fn_bos(args.batch_size).to(device)
        fixed_boson_nl_z = sampler_fn_nl(args.batch_size).to(device)
    # quick generation test (only active branches)
    shapes_msg = []
    if args.latent_space in ('base', 'all'):
        e_base, n_base = G_base(fixed_baseline_z)
        shapes_msg.append(f"Baseline -> edge: {e_base.shape}, node: {n_base.shape}")
    if args.latent_space in ('bos', 'all'):
        e_boson, n_boson = G_bos(fixed_boson_z)
        shapes_msg.append(f"Boson -> edge: {e_boson.shape}, node: {n_boson.shape}")
    if args.latent_space in ('nl', 'all'):
        e_boson_nl, n_boson_nl = G_nl(fixed_boson_nl_z)
        shapes_msg.append(f"Boson-NL -> edge: {e_boson_nl.shape}, node: {n_boson_nl.shape}")
    print(" | ".join(shapes_msg))
    # initialize training timers (ms)

    for epoch in range(1, args.epochs+1):
        print(f"Starting epoch {epoch}/{args.epochs}")
        # reset per‑epoch timers
        baseline_time_ms = boson_time_ms = boson_nl_time_ms = 0.0
        reward_sum = 0.0
        reward_count = 0
        # Track losses for each branch
        d_loss_base = g_loss_base = 0.0
        d_loss_bos = g_loss_bos = 0.0
        d_loss_nl = g_loss_nl = 0.0
        for batch_idx, (real_adj, real_feats) in enumerate(loader):
            if batch_idx % 10 == 0:
                print(f" Epoch {epoch}, batch {batch_idx+1}/{len(loader)}")
            real_adj = real_adj.to(device, non_blocking=True)
            real_feats = real_feats.to(device, non_blocking=True)

            # --- Multi-branch training ---
            # Each tuple: (G_i, D_i, R_i, sampler_fn_i, optG_i, optD_i, optR_i, loss accumulators)
            branches_all = [
                (G_base, D_base, R_base, sampler_fn_base, optG_base, optD_base, optR_base, 'base'),
                (G_bos,  D_bos,  R_bos,  sampler_fn_bos,  optG_bos,  optD_bos,  optR_bos,  'bos'),
                (G_nl,   D_nl,   R_nl,   sampler_fn_nl,   optG_nl,   optD_nl,   optR_nl,   'nl')
            ]
            branches = [b for b in branches_all
                        if args.latent_space == 'all' or b[7] == args.latent_space]
            losses = {}
            for G_i, D_i, R_i, sampler_fn_i, optG_i, optD_i, optR_i, label in branches:
                # ----- Discriminator (WGAN-GP) -----
                z = sampler_fn_i(args.batch_size).to(device, non_blocking=True)
                t_train = time.perf_counter()
                fake_adj, fake_feats = G_i(z, hard=False)   # soft probs for D/GP
                d_real = D_i(real_adj,  real_feats).clamp(-10, 10).mean()
                d_fake = D_i(fake_adj.detach(), fake_feats.detach()).clamp(-10, 10).mean()
                # select branch-specific GP weight
                if label == 'base':
                    gp_weight = args.lambda_gp_base
                elif label == 'bos':
                    gp_weight = args.lambda_gp_bos
                else:
                    gp_weight = args.lambda_gp_nl
                gp = compute_gradient_penalty(
                        D_i,
                        (real_adj, real_feats),
                        (fake_adj, fake_feats),
                        device,
                        gp_weight)
                # --- NaN guards: replace NaNs/Infs in discriminator components ---
                d_real = torch.nan_to_num(d_real, nan=0.0, posinf=1e6, neginf=-1e6)
                d_fake = torch.nan_to_num(d_fake, nan=0.0, posinf=1e6, neginf=-1e6)
                gp     = torch.nan_to_num(gp,     nan=0.0, posinf=1e6, neginf=-1e6)
                d_loss = d_fake - d_real + gp
                if torch.isnan(d_loss) or torch.isinf(d_loss):
                    print(f"[Skip] Invalid d_loss (NaN/Inf) for branch '{label}' at epoch {epoch}, batch {batch_idx}")
                    losses[f'd_loss_{label}'] = float('nan')
                else:
                    optD_i.zero_grad()
                    d_loss.backward()
                    clip_grad_norm_(D_i.parameters(), max_norm=1.0)
                    optD_i.step()
                    losses[f'd_loss_{label}'] = d_loss.item()
                elapsed_ms = (time.perf_counter() - t_train) * 1e3
                if label == 'base':
                    baseline_time_ms += elapsed_ms
                elif label == 'bos':
                    boson_time_ms += elapsed_ms
                else:  # 'nl'
                    boson_nl_time_ms += elapsed_ms
                # ----- Reward network (supervise on real data) -----
                with torch.no_grad():
                    metrics_batch = metrics.compute(real_adj, real_feats)
                    true_reward = metrics_batch['reward']
                    reward_sum += true_reward.item() if torch.is_tensor(true_reward) else float(true_reward)
                    reward_count += 1
                # update reward network
                r_pred = R_i(real_adj, real_feats)
                r_loss = ((r_pred - true_reward) ** 2).mean()
                optR_i.zero_grad()
                r_loss.backward()
                optR_i.step()
                losses[f'r_loss_{label}'] = r_loss.item()
                # ----- Generator -----
                if batch_idx % args.n_critic == 0:
                    z = sampler_fn_i(args.batch_size).to(device, non_blocking=True)
                    t_train = time.perf_counter()
                    fake_adj, fake_feats = G_i(z, hard=False)
                    d_fake = D_i(fake_adj, fake_feats).mean()
                    r_fake = R_i(fake_adj, fake_feats).mean()
                    # select branch-specific RL weight
                    if label == 'base':
                        rl_weight = args.lambda_rl_base
                    elif label == 'bos':
                        rl_weight = args.lambda_rl_bos
                    else:
                        rl_weight = args.lambda_rl_nl
                    g_loss = -d_fake + rl_weight * (-r_fake)
                    if torch.isnan(g_loss) or torch.isinf(g_loss):
                        print(f"[Skip] Invalid g_loss (NaN/Inf) for branch '{label}' at epoch {epoch}, batch {batch_idx}")
                    else:
                        optG_i.zero_grad()
                        g_loss.backward()
                        clip_grad_norm_(G_i.parameters(), max_norm=1.0)
                        optG_i.step()
                        elapsed_ms = (time.perf_counter() - t_train) * 1e3
                        if label == 'base':
                            baseline_time_ms += elapsed_ms
                        elif label == 'bos':
                            boson_time_ms += elapsed_ms
                        else:  # 'nl'
                            boson_nl_time_ms += elapsed_ms
                        losses[f'g_loss_{label}'] = g_loss.item()
            for lbl in ('base', 'bos', 'nl'):
                losses.setdefault(f'd_loss_{lbl}', float('nan'))
                losses.setdefault(f'g_loss_{lbl}', float('nan'))
            # Save for print outside batch loop
            d_loss_base = losses['d_loss_base']
            g_loss_base = losses.get('g_loss_base', g_loss_base)
            d_loss_bos  = losses['d_loss_bos']
            g_loss_bos  = losses.get('g_loss_bos', g_loss_bos)
            d_loss_nl   = losses['d_loss_nl']
            g_loss_nl   = losses.get('g_loss_nl', g_loss_nl)

        msg = []
        if args.latent_space in ('base', 'all'):
            msg.append(f"Baseline D: {d_loss_base:.4f}, G: {g_loss_base:.4f}")
        if args.latent_space in ('bos', 'all'):
            msg.append(f"Boson D: {d_loss_bos:.4f}, G: {g_loss_bos:.4f}")
        if args.latent_space in ('nl', 'all'):
            msg.append(f"Boson-NL D: {d_loss_nl:.4f}, G: {g_loss_nl:.4f}")
        print(f"Epoch {epoch} | " + ' | '.join(msg))
        # ---- Evaluate generator reward on a fixed latent batch ----
        with torch.no_grad():
            gen_reward_base = gen_reward_bos = gen_reward_nl = float('nan')

            if args.latent_space in ('base', 'all'):
                fake_adj_eval, fake_feats_eval = G_base(fixed_baseline_z)
                gen_reward_base = metrics.compute(fake_adj_eval, fake_feats_eval)['reward']
            if args.latent_space in ('bos', 'all'):
                fake_adj_eval_b, fake_feats_eval_b = G_bos(fixed_boson_z)
                gen_reward_bos = metrics.compute(fake_adj_eval_b, fake_feats_eval_b)['reward']
            if args.latent_space in ('nl', 'all'):
                fake_adj_eval_nl, fake_feats_eval_nl = G_nl(fixed_boson_nl_z)
                gen_reward_nl = metrics.compute(fake_adj_eval_nl, fake_feats_eval_nl)['reward']

            if args.latent_space == 'base':
                reward_val = gen_reward_base
            elif args.latent_space == 'bos':
                reward_val = gen_reward_bos
            elif args.latent_space == 'nl':
                reward_val = gen_reward_nl
            else:  # 'all' → default to baseline reward
                reward_val = gen_reward_base
        if getattr(args, 'wandb', False) and wandb.run is not None:
            relative_time = (baseline_time_ms + boson_time_ms + boson_nl_time_ms) / 3
            avg_reward = reward_sum / reward_count if reward_count else 0.0
            wandb.log({
                'seed': seed,
                'epoch': epoch,
                'baseline_train_ms': baseline_time_ms,
                'boson_train_ms': boson_time_ms,
                'boson_nl_train_ms': boson_nl_time_ms,
                'd_loss_base': d_loss_base,
                'g_loss_base': g_loss_base,
                'd_loss_bos':  d_loss_bos,
                'g_loss_bos':  g_loss_bos,
                'd_loss_nl':   d_loss_nl,
                'g_loss_nl':   g_loss_nl,
                'Relative Time (Process)': relative_time,
                # 'avg_reward': avg_reward,  # Removed as per instructions
                'reward': reward_val,
                'gen_reward_base': gen_reward_base,
                'gen_reward_bos':  gen_reward_bos,
                'gen_reward_nl':   gen_reward_nl,
            })
        if writer is not None:
            # compute W1 distances for logging (per latent space)
            real_batch = next(iter(loader))
            real_adj_batch, real_feats_batch = real_batch
            real_adj_batch = real_adj_batch.to(device)
            real_feats_batch = real_feats_batch.to(device)

            # Compute W1 for each latent space
            w1_base = w1_bos = w1_nl = float('nan')
            with torch.no_grad():
                if args.latent_space in ('base', 'all'):
                    real_score_base = D_base(real_adj_batch, real_feats_batch).mean().item()
                    fake_adj_eval, fake_feats_eval = G_base(fixed_baseline_z)
                    fake_score_base = D_base(fake_adj_eval, fake_feats_eval).mean().item()
                    w1_base = real_score_base - fake_score_base
                if args.latent_space in ('bos', 'all'):
                    real_score_bos = D_bos(real_adj_batch, real_feats_batch).mean().item()
                    fake_adj_eval_b, fake_feats_eval_b = G_bos(fixed_boson_z)
                    fake_score_bos = D_bos(fake_adj_eval_b, fake_feats_eval_b).mean().item()
                    w1_bos = real_score_bos - fake_score_bos
                if args.latent_space in ('nl', 'all'):
                    real_score_nl = D_nl(real_adj_batch, real_feats_batch).mean().item()
                    fake_adj_eval_nl, fake_feats_eval_nl = G_nl(fixed_boson_nl_z)
                    fake_score_nl = D_nl(fake_adj_eval_nl, fake_feats_eval_nl).mean().item()
                    w1_nl = real_score_nl - fake_score_nl

            # Compute metrics for each latent space
            validity_base = qed_base = logp_base = sa_base = float('nan')
            validity_bos  = qed_bos  = logp_bos  = sa_bos  = float('nan')
            validity_nl   = qed_nl   = logp_nl   = sa_nl   = float('nan')
            with torch.no_grad():
                if args.latent_space in ('base', 'all'):
                    metrics_base = metrics.compute(fake_adj_eval, fake_feats_eval)
                    validity_base = float(metrics_base['validity'])
                    qed_base      = float(metrics_base['qed'])
                    logp_base     = float(metrics_base['logp'])
                    sa_base       = float(metrics_base['sa'])
                if args.latent_space in ('bos', 'all'):
                    metrics_bos = metrics.compute(fake_adj_eval_b, fake_feats_eval_b)
                    validity_bos = float(metrics_bos['validity'])
                    qed_bos      = float(metrics_bos['qed'])
                    logp_bos     = float(metrics_bos['logp'])
                    sa_bos       = float(metrics_bos['sa'])
                if args.latent_space in ('nl', 'all'):
                    metrics_nl = metrics.compute(fake_adj_eval_nl, fake_feats_eval_nl)
                    validity_nl = float(metrics_nl['validity'])
                    qed_nl      = float(metrics_nl['qed'])
                    logp_nl     = float(metrics_nl['logp'])
                    sa_nl       = float(metrics_nl['sa'])

            writer.writerow({
                'seed': seed,
                'epoch': epoch,
                'baseline_train_ms': baseline_time_ms,
                'boson_train_ms': boson_time_ms,
                'boson_nl_train_ms': boson_nl_time_ms,
                'd_loss_base': d_loss_base,
                'g_loss_base': g_loss_base,
                'd_loss_bos': d_loss_bos,
                'g_loss_bos': g_loss_bos,
                'd_loss_nl': d_loss_nl,
                'g_loss_nl': g_loss_nl,
                'validity_base': validity_base,
                'qed_base': qed_base,
                'logp_base': logp_base,
                'sa_base': sa_base,
                'reward_base': float(gen_reward_base),
                'validity_bos': validity_bos,
                'qed_bos': qed_bos,
                'logp_bos': logp_bos,
                'sa_bos': sa_bos,
                'reward_bos': float(gen_reward_bos),
                'validity_nl': validity_nl,
                'qed_nl': qed_nl,
                'logp_nl': logp_nl,
                'sa_nl': sa_nl,
                'reward_nl': float(gen_reward_nl),
                'w1_base': w1_base,
                'w1_bos': w1_bos,
                'w1_nl': w1_nl,
            })

            # Save per-epoch branch-specific checkpoints
            for label, G_i, D_i, R_i in [
                ('baseline', G_base, D_base, R_base),
                ('boson',    G_bos,  D_bos,  R_bos),
                ('nonlinear',G_nl,   D_nl,   R_nl)
            ]:
                dir_path = os.path.join(args.weights_dir, label)
                os.makedirs(dir_path, exist_ok=True)
                ckpt = {
                    'seed': seed,
                    'epoch': epoch,
                    'G': G_i.state_dict(),
                    'D': D_i.state_dict(),
                    'R': R_i.state_dict(),
                    'args': vars(args)
                }
                filename = f"seed{seed}_epoch{epoch}.pt"
                torch.save(ckpt, os.path.join(dir_path, filename))
                print(f"Saved {label} checkpoint -> {os.path.join(dir_path, filename)}")

    # After training epochs, log final accuracy once
    if getattr(args, 'wandb', False) and wandb.run is not None:
        with torch.no_grad():
            z_final = sampler_fn_base(args.batch_size).to(device)
            fake_adj_final, fake_feats_final = G_base(z_final)
            vm_final = metrics.compute(fake_adj_final, fake_feats_final)
            final_accuracy = vm_final.get('validity', 0.0)
            final_reward = vm_final.get('reward', 0.0)
        wandb.log({'accuracy': final_accuracy, 'reward': final_reward})

    # ---- Save weights once training for this seed is finished ----
    os.makedirs(args.weights_dir, exist_ok=True)
    checkpoint = {
        'seed': seed,
        'epoch': epoch,
        'G_base': G_base.state_dict(),
        'D_base': D_base.state_dict(),
        'R_base': R_base.state_dict(),
        'G_bos':  G_bos.state_dict(),
        'D_bos':  D_bos.state_dict(),
        'R_bos':  R_bos.state_dict(),
        'G_nl':   G_nl.state_dict(),
        'D_nl':   D_nl.state_dict(),
        'R_nl':   R_nl.state_dict(),
        'args':   vars(args),
    }
    ckpt_path = os.path.join(args.weights_dir, f'weights_seed{seed}.pt')
    torch.save(checkpoint, ckpt_path)
    print(f"Saved checkpoint -> {ckpt_path}")

    if getattr(args, 'wandb', False) and wandb.run is not None:
        wandb.save(ckpt_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', type=str, required=True)
    p.add_argument('--subset', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--z-dim', '--z_dim', type=int, default=10)
    # REMOVED: hidden-dim, tau, lr
    # p.add_argument('--hidden-dim', '--hidden_dim', type=int, default=128)
    # p.add_argument('--tau', type=float, default=1.0)
    # p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--benchmark', action='store_true',
                   help='Measure and print sampler latencies')
    p.add_argument('--latent-space', '--latent_space', type=str,
               choices=['base', 'bos', 'nl', 'all'],
               default='base',
               help="Which latent space to train: "
                    "'base' = Gaussian, 'bos' = linear boson, "
                    "'nl' = nonlinear boson, 'all' = train all three")
    p.add_argument('--n-critic', '--n_critic', type=int, default=5,
                   help='Number of discriminator updates per generator update')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lambda-gp', '--lambda_gp', type=float, default=10.0)
    p.add_argument('--lambda-rl', '--lambda_rl', type=float, default=1.0)
    p.add_argument('--lambda-gp-base', type=float, default=None, help='Gradient penalty weight for base latent space')
    p.add_argument('--lambda-gp-bos',  type=float, default=None, help='Gradient penalty weight for boson latent space')
    p.add_argument('--lambda-gp-nl',   type=float, default=None, help='Gradient penalty weight for nonlinear boson latent space')
    p.add_argument('--lambda-rl-base', type=float, default=None, help='RL weight for base latent space')
    p.add_argument('--lambda-rl-bos',  type=float, default=None, help='RL weight for boson latent space')
    p.add_argument('--lambda-rl-nl',   type=float, default=None, help='RL weight for nonlinear boson latent space')
    # reward weights
    p.add_argument('--w-valid', '--w_valid', type=float, default=0.1)
    p.add_argument('--w-qed', '--w_qed', type=float, default=0.3)
    p.add_argument('--w-logp', '--w_logp', type=float, default=0.3)
    p.add_argument('--w-sa', '--w_sa', type=float, default=0.3)
    # New arguments for seeds and output CSV
    p.add_argument('--seeds', type=int, nargs='+', default=[0], help='List of random seeds to run')
    p.add_argument('--output-csv', type=str, default='results.csv', help='Path to output CSV file')
    p.add_argument('--weights-dir', type=str, default='weights',
                   help='Directory to save model weights (one file per seed)')
    p.add_argument('--wandb', action='store_true',
               help='Enable Weights & Biases logging')
    p.add_argument('--wandb-project', type=str,
                default='nonlinear-photonic-sampler',
                help='W&B project name')
    p.add_argument('--wandb-name', type=str, default=None,
                help='Optional W&B run name')
    # Add new branch-specific hyperparameters
    p.add_argument('--hidden-dim-base', type=int, default=128, help='Hidden dimension for base latent space')
    p.add_argument('--hidden-dim-bos', type=int, default=128, help='Hidden dimension for boson latent space')
    p.add_argument('--hidden-dim-nl', type=int, default=128, help='Hidden dimension for nonlinear boson latent space')
    p.add_argument('--tau-base', type=float, default=1.0, help='Tau for Gumbel-Softmax in base latent space')
    p.add_argument('--tau-bos', type=float, default=1.0, help='Tau for Gumbel-Softmax in boson latent space')
    p.add_argument('--tau-nl', type=float, default=1.0, help='Tau for Gumbel-Softmax in nonlinear boson latent space')
    p.add_argument('--hidden-dim', '--hidden_dim', type=int, default=None, help='Global hidden dimension override for chosen latent space(s)')
    p.add_argument('--tau', type=float, default=None, help='Global tau override for chosen latent space(s)')
    p.add_argument('--lr', type=float, default=None, help='Global learning rate override for chosen latent space(s)')
    p.add_argument('--lr-base', type=float, default=1e-3, help='Learning rate for base latent space')
    p.add_argument('--lr-bos', type=float, default=1e-3, help='Learning rate for boson latent space')
    p.add_argument('--lr-nl', type=float, default=1e-3, help='Learning rate for nonlinear boson latent space')
    # Add branch-specific latent dim arguments
    p.add_argument('--z-dim-base', type=int, default=None, help='Latent dimension for base latent space')
    p.add_argument('--z-dim-bos', type=int, default=None, help='Latent dimension for boson latent space')
    p.add_argument('--z-dim-nl', type=int, default=None, help='Latent dimension for nonlinear boson latent space')
    args = p.parse_args()
    # Default branch-specific GP and RL weights to global values if not provided
    if args.lambda_gp_base is None: args.lambda_gp_base = args.lambda_gp
    if args.lambda_gp_bos  is None: args.lambda_gp_bos  = args.lambda_gp
    if args.lambda_gp_nl   is None: args.lambda_gp_nl   = args.lambda_gp
    if args.lambda_rl_base is None: args.lambda_rl_base = args.lambda_rl
    if args.lambda_rl_bos  is None: args.lambda_rl_bos  = args.lambda_rl
    if args.lambda_rl_nl   is None: args.lambda_rl_nl   = args.lambda_rl
    # Default branch-specific latent dims to global z-dim if not provided
    if args.z_dim_base is None:
        args.z_dim_base = args.z_dim
    if args.z_dim_bos is None:
        args.z_dim_bos = args.z_dim
    if args.z_dim_nl is None:
        args.z_dim_nl = args.z_dim

    # Apply global lr override to active latent space(s)
    if args.lr is not None:
        if args.latent_space in ('base', 'all'):
            args.lr_base = args.lr
        if args.latent_space in ('bos', 'all'):
            args.lr_bos = args.lr
        if args.latent_space in ('nl', 'all'):
            args.lr_nl = args.lr

    # Apply global hidden-dim override to active latent space(s)
    if args.hidden_dim is not None:
        if args.latent_space in ('base', 'all'):
            args.hidden_dim_base = args.hidden_dim
        if args.latent_space in ('bos', 'all'):
            args.hidden_dim_bos = args.hidden_dim
        if args.latent_space in ('nl', 'all'):
            args.hidden_dim_nl = args.hidden_dim

    # Apply global tau override to active latent space(s)
    if args.tau is not None:
        if args.latent_space in ('base', 'all'):
            args.tau_base = args.tau
        if args.latent_space in ('bos', 'all'):
            args.tau_bos = args.tau
        if args.latent_space in ('nl', 'all'):
            args.tau_nl = args.tau

    if args.wandb:
        wandb.init(project=args.wandb_project,
                name=args.wandb_name,
                config=vars(args))

    if args.seeds and len(args.seeds) > 1:
        with open(args.output_csv, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile,fieldnames=[
                'seed', 'epoch',
                'baseline_train_ms', 'boson_train_ms', 'boson_nl_train_ms',
                'd_loss_base', 'g_loss_base',
                'd_loss_bos',  'g_loss_bos',
                'd_loss_nl',   'g_loss_nl',
                'validity_base', 'qed_base', 'logp_base', 'sa_base', 'reward_base',
                'validity_bos',  'qed_bos',  'logp_bos',  'sa_bos',  'reward_bos',
                'validity_nl',   'qed_nl',   'logp_nl',   'sa_nl',   'reward_nl',
                'w1_base', 'w1_bos', 'w1_nl',
            ])
            writer.writeheader()
            for seed in args.seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                print(f"Running with seed {seed}")
                train(args, seed, writer)
        print(f"Results written to {args.output_csv}")
    else:
        # Single run – still persist per‑epoch metrics to CSV
        with open(args.output_csv, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                'seed', 'epoch',
                'baseline_train_ms', 'boson_train_ms', 'boson_nl_train_ms',
                'd_loss_base', 'g_loss_base',
                'd_loss_bos',  'g_loss_bos',
                'd_loss_nl',   'g_loss_nl',
                'validity_base', 'qed_base', 'logp_base', 'sa_base', 'reward_base',
                'validity_bos',  'qed_bos',  'logp_bos',  'sa_bos',  'reward_bos',
                'validity_nl',   'qed_nl',   'logp_nl',   'sa_nl',   'reward_nl',
                'w1_base', 'w1_bos', 'w1_nl',
            ])
            writer.writeheader()

            # Use the first (or only) seed provided; if none, default to 0
            seed_val = args.seeds[0] if args.seeds else 0
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_val)

            print(f"Running with seed {seed_val}")
            train(args, seed_val, writer)
        print(f"Results written to {args.output_csv}")