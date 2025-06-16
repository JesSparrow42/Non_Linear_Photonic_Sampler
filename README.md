# Non-Linear Photonic Sampler  
*A boson-sampling simulator with quantum‑aware latent generators for GAN / VAE / Graph‑VAE pipelines*

[![Python ≥3.9](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

---

## Overview
`bosonsampler-wrapper` is an internal research toolbox that provides  

* **Fast photonic boson‑sampling simulation** (linear & quantum‑dot–non‑linear sources)  
* **A differentiable PyTorch interface** (`BosonSamplerTorch`) for end‑to‑end optimisation  
* **Physics‑aware latent vectors** (`BosonLatentGenerator`) that plug directly into generative models  
* **Reference GAN, VAE and Graph‑VAE training pipelines** for MNIST, PET/CT medical‑imaging and QM9 molecular‑graph data  
* **Utility modules** for DICOM visualisation, molecular metrics, custom losses, plotting and evaluation  

---

## Repository layout

```
NONLINEAR-PHOTONIC-SAMPLER/
├─ bosonsampler/                 # Core simulation engine
│  ├─ core.py                    # permanents, Monte‑Carlo, torch wrappers
│  ├─ __init__.py | __version__.py
├─ tests/
│  ├─ mnist/                     # WGAN‑GP demo
│  ├─ pet_ct/                    # 3‑D VAE & GAN pipelines
│  └─ qm9/                       # Graph‑VAE & molecular score-based models based on MolGAN
│      ├─ dataset.py                   # dense graph preprocessing & loader
│      ├─ sparse_molecular_dataset.py  # sparse COO representation
│      ├─ layers.py                    # GNN / R‑GCN layers
│      ├─ models.py                    # GVAE and score network
│      ├─ train.py                     # training script
│      ├─ evaluate.py                  # QED, logP, SA, validity 
│      ├─ molecular_metrics.py         # metric helpers + ChemNet wrapper
│      ├─ plot_fcd.py                  # plot FréchetChemNet progress
│      ├─ utils.py                     # misc helpers
│      └─ download_dataset.sh          # fetch & preprocess QM9
├─ plotting_tools/               # CSV → matplotlib scripts
├─ training_data/                # (empty) – drop your datasets here
├─ pyproject.toml                # Build & dependency management
└─ README.md
```

---

## Installation

```bash
git clone <internal-git-url>/nonlinear-photonic-sampler.git
cd nonlinear-photonic-sampler
python -m venv .venv && source .venv/bin/activate   # optional
pip install --upgrade pip
pip install -e ".[dev]"   # pulls in pytest / black / ruff, etc.
```

> **Note**  
> `numba` is installed only on x86‑64 Linux & Windows because the Ryser JIT kernel currently does not compile on Apple Silicon.

---

## Quick‑start

### 1 – Simulate a boson‑sampling experiment

```python
import torch
from bosonsampler import BosonSamplerTorch, generate_unitary

m, num_sources = 6, 4
unitary = generate_unitary(m)

sampler = BosonSamplerTorch(
    m, num_sources,
    num_loops=1_000,
    input_loss=0.02, coupling_efficiency=0.9,
    detector_inefficiency=0.1, mu=0.95,
    temporal_mismatch=0.05, spectral_mismatch=0.05,
    arrival_time_jitter=0.01, bs_loss=0.02, bs_jitter=0.005,
    phase_noise_std=0.01, systematic_phase_offset=0.0,
    mode_loss=0.01, unitary=unitary,
)

prob, mode_dist = sampler(batch_size=128)
print(mode_dist)      # latent vector (m,)
```

### 2 – Train a WGAN‑GP on MNIST with a bosonic latent generator

```bash
python tests/MNIST/train.py \
       --epochs 200 \
       --save_dir model_checkpoints/mnist
```

### 3 – Train a VAE on PET/CT volumes

```bash
python utils/train_vae.py \
       --data_dir training_data/PET_CT \
       --checkpoint_dir model_checkpoints/pet_ct
```

### 4 – Train a Graph‑VAE on QM9 molecules

```bash
# one‑time download (≈115 MB)
bash experiments/qm9/download_dataset.sh

python experiments/qm9/train.py \
       --model gvae \
       --epochs 300 \
       --lat_dim 16 \
       --boson_latent True \
       --save_dir checkpoints/qm9_gvae
```

Evaluate molecular validity / uniqueness / FréchetChemNet:

```bash
python experiments/qm9/evaluate.py \
       --checkpoint checkpoints/qm9_gvae/best.pt
```

Learning curves are logged to CSV files; visualise them via

```bash
python plotting_tools/plot_csv.py \
       --file generator_losses_nonlinearity.csv
```

---

## Core API

| Component                              | Description                                                         |
|----------------------------------------|---------------------------------------------------------------------|
| `bosonsampler.core.boson_sampling_simulation` | NumPy Monte‑Carlo engine                                            |
| `BosonSamplerTorch`                    | `nn.Module` wrapper, fully differentiable                            |
| `BosonLatentGenerator`                 | Converts mode distribution → fixed‑size latent vector               |
| `gan_module.Generator` / `Discriminator` | U‑Net‑inspired models for WGAN‑GP                                   |
| `vae_module.VAE`                       | Variational auto‑encoder with a bosonic prior                       |
| `utils.data_loader.create_data_loader` | Zero‑copy DICOM → torch `DataLoader`                                |
| `qm9.models.GVAE` / `ScoreModel`          | Graph VAE & score‑based molecular generative model        |
| `qm9.molecular_metrics.*`                 | Validity, uniqueness, FréchetChemNet, KL, …               |

---

## Testing

```bash
pytest -q
```

---

## Development workflow

```bash
# style
ruff check . --fix
black .

# type‑checking
mypy bosonsampler
```

---

## Internal citation

If this toolkit informs internal reports or papers, please acknowledge:

```
Nørregaard, O. *Non‑Linear Photonic Boson‑Sampling Simulator & Generative‑Model Toolkit* (2025).
```
