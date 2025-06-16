#!/usr/bin/env python3
# visualize_gan_training.py
#
# Plot mean ± SEM of losses, cumulative latencies,
# and a loss-vs-latency curve per branch.
#
# ▸ Depends only on pandas and matplotlib

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np                    # just for sqrt

# ────────────────────────────────────────────────────────────
# 1.  Load and clean
# ────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__),
                        "/Users/olivernorregaard/Documents/GitHub/nonlinear-photonic-sampler/generator_losses_nonlinearity.csv")

df = pd.read_csv(CSV_PATH)
main = df[df["Phase"] == "Main Training"].copy()

numeric_cols = [
    "Baseline_Loss", "Boson_Gen_Loss", "Boson_Gen_Nonlinear_Loss",
    "Baseline_Latency_ms", "Boson_Latency_ms", "Boson_NL_Latency_ms",
]
main[numeric_cols] = main[numeric_cols].apply(
    pd.to_numeric, errors="coerce"
)

# ────────────────────────────────────────────────────────────
# 2.  Mean & SEM per epoch across seeds
# ────────────────────────────────────────────────────────────
N_runs = main.groupby("Epoch").size()           # how many seeds contributed
stats = (
    main.groupby("Epoch")[numeric_cols]
    .agg(["mean", "std"])
    .sort_index()
)
epochs = stats.index.values

def mean(metric):
    return stats[(metric, "mean")]

def sem(metric):
    return stats[(metric, "std")] / np.sqrt(N_runs)

# ────────────────────────────────────────────────────────────
# Loss curves   (mean ± SEM)
# ────────────────────────────────────────────────────────────
plt.figure()
for metric, label in [
    ("Baseline_Loss", "Gaussian"),
   # ("Boson_Gen_Loss", "Boson"),
    ("Boson_Gen_Nonlinear_Loss", "Boson NL"),
]:
    m, e = mean(metric), sem(metric)
    plt.plot(epochs, m, label=label)
    plt.fill_between(epochs, m - e, m + e, alpha=0.20)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.tight_layout()

# ────────────────────────────────────────────────────────────
# Latency curves   (mean ± SEM)
# ────────────────────────────────────────────────────────────
plt.figure()
for metric, label in [
    ("Baseline_Latency_ms", "Gaussian"),
    ("Boson_Latency_ms", "Boson"),
    ("Boson_NL_Latency_ms", "Boson NL"),
]:
    m, e = mean(metric), sem(metric)
    plt.plot(epochs, m, label=label)
    plt.fill_between(epochs, m - e, m + e, alpha=0.20)
plt.xlabel("Epoch"); plt.ylabel("Cumulative latency (ms)")
plt.legend(); plt.tight_layout()

# ────────────────────────────────────────────────────────────
# Loss vs Latency scatter / line
# ────────────────────────────────────────────────────────────
plt.figure()
for loss_m, lat_m, label in [
    ("Baseline_Loss",          "Baseline_Latency_ms",         "Gaussian"),
    ("Boson_Gen_Loss",         "Boson_Latency_ms",            "Boson"),
    ("Boson_Gen_Nonlinear_Loss","Boson_NL_Latency_ms",        "Boson NL"),
]:
    plt.plot(mean(lat_m), mean(loss_m), label=label)
plt.xlabel("Cumulative latency (ms)")
plt.ylabel("Generator loss")
plt.legend()
plt.tight_layout()

# ────────────────────────────────────────────────────────────
# Cumulative latency vs Epoch (duplicate view, mean ± SEM)
# ────────────────────────────────────────────────────────────
plt.figure()
for metric, label in [
    ("Baseline_Latency_ms", "Gaussian"),
    ("Boson_Latency_ms",    "Boson"),
    ("Boson_NL_Latency_ms", "Boson NL"),
]:
    m, e = mean(metric), sem(metric)
    plt.plot(epochs, m, label=label)
    plt.fill_between(epochs, m - e, m + e, alpha=0.20)
plt.xlabel("Epoch")
plt.ylabel("Cumulative latency (ms)")
plt.legend()
plt.tight_layout()

plt.show()