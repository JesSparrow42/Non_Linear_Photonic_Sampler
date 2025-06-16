#!/usr/bin/env python3
import argparse
import os
import pickle
import numpy as np
from bosonsampler.core import generate_unitary, sample_distribution, sample_distribution_nl_torch

def main():
    parser = argparse.ArgumentParser(
        description="Precompute and save a set of boson-sampling distributions."
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path to a .pkl file where the distributions will be saved."
    )
    parser.add_argument(
        "--m",
        type=int,
        required=True,
        help="Number of modes (length of input_state)."
    )
    parser.add_argument(
        "--input-state",
        type=int,
        nargs="+",
        required=True,
        help="Fock input state as a list of length m (e.g. 1 0 1 0 ...)."
    )
    parser.add_argument(
        "--num-distributions",
        type=int,
        default=20000,
        help="How many random-unitary distributions to generate."
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of Monte Carlo samples per distribution."
    )
    parser.add_argument(
        "--input-loss",
        type=float,
        default=0.0,
        help="Per-photon source-loss probability."
    )
    parser.add_argument(
        "--bs-loss",
        type=float,
        default=1.0,
        help="Beam-splitter loss factor."
    )
    parser.add_argument(
        "--bs-jitter",
        type=float,
        default=0.0,
        help="Beam-splitter amplitude-noise (jitter)."
    )
    parser.add_argument(
        "--phase-noise-std",
        type=float,
        default=0.0,
        help="Phase-noise standard deviation."
    )
    parser.add_argument(
        "--mode-loss",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Per-mode loss factors (length m). "
            "If omitted, no per-mode loss is applied."
        )
    )
    parser.add_argument(
        "--nonlinear",
        action="store_true",
        help="Generate nonlinear boson-sampling distributions using sample_distribution_nl_torch."
    )
    parser.add_argument(
        "--detuning",
        type=float,
        default=0.0,
        help="Detuning parameter for nonlinear sampling."
    )
    parser.add_argument(
        "--pulse-bw",
        type=float,
        default=0.0,
        help="Pulse bandwidth for nonlinear sampling."
    )
    parser.add_argument(
        "--qd-linewidth",
        type=float,
        default=1.0,
        help="Quantum dot linewidth for nonlinear sampling."
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.0,
        help="Phi linear parameter for nonlinear sampling."
    )
    parser.add_argument(
        "--dark-count-rate",
        type=float,
        default=0.0,
        help="Dark count rate for nonlinear sampling."
    )
    args = parser.parse_args()

    # Validate input_state length
    if len(args.input_state) != args.m:
        raise ValueError("Length of input_state must equal m")
    # No additional validation needed; default values cover nonlinear parameters

    # Convert input_state to numpy array once
    input_arr = np.array(args.input_state, dtype=int)

    # Prepare container for all distributions
    all_distributions = []

    for i in range(args.num_distributions):
        U = generate_unitary(
            args.m,
            args.bs_loss,
            args.bs_jitter,
            args.phase_noise_std,
            0.0,  # no systematic phase offset during precompute
            args.mode_loss
        )
        if args.nonlinear:
            # Generate nonlinear distribution using sample_distribution_nl_torch
            states_t, probs_t = sample_distribution_nl_torch(
                U,
                input_arr,
                device="cpu",
                n_samples=args.n_samples,
                detuning=args.detuning,
                pulse_bw=args.pulse_bw,
                qd_linewidth=args.qd_linewidth,
                phi_linear=args.phi,
                input_loss=args.input_loss,
                dark_count_rate=args.dark_count_rate
            )
            states_list = states_t.detach().cpu().tolist()
            probs_list = probs_t.detach().cpu().tolist()
        else:
            dist = sample_distribution(
                U,
                input_arr,
                n_samples=args.n_samples,
                input_loss=args.input_loss
            )
            states_list = list(dist.keys())
            probs_list = list(dist.values())
        all_distributions.append((states_list, probs_list))

        # Optional: print progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{args.num_distributions} distributions")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Save to pickle
    with open(args.output_file, "wb") as f:
        pickle.dump(all_distributions, f)

    print(f"Saved {args.num_distributions} distributions to {args.output_file}")

if __name__ == "__main__":
    main()