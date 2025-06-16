import os
import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.distributions import Distribution
from .utils.vae_utils import BosonPrior, ReparameterizedDiagonalGaussian
from ptseries.models import PTGenerator
from bosonsampler import BosonLatentGenerator, BosonSamplerTorch

from torch.distributions import Normal

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_shape: torch.Size, latent_features: int, boson_sampler_params: dict = None):
        super().__init__()
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = int(np.prod(input_shape))  # e.g. 128x128 -> 16384

        self.encoder = nn.Sequential(
            nn.Linear(self.observation_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 2 * latent_features)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_features, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, self.observation_features)
        )

        self.boson_sampler_params = boson_sampler_params
        self.boson_sampler = None
        if boson_sampler_params is not None:
            # Choose PTSeries prior if tbi_params key is present, else use BosonSamplerTorch
            if 'tbi_params' in boson_sampler_params:
                self.boson_sampler = PTGenerator(**boson_sampler_params)
            else:
                # Instantiate BosonSamplerTorch with provided parameters
                bs_module = BosonSamplerTorch(
                    m=self.latent_features,
                    num_sources=boson_sampler_params.get("num_sources"),
                    num_loops=boson_sampler_params.get("num_loops", 1),
                    input_loss=boson_sampler_params.get("input_loss", 0.0),
                    coupling_efficiency=boson_sampler_params.get("coupling_efficiency", 1.0),
                    detector_inefficiency=boson_sampler_params.get("detector_inefficiency", 1.0),
                    mu=boson_sampler_params.get("mu", 1.0),
                    temporal_mismatch=boson_sampler_params.get("temporal_mismatch", 0.0),
                    spectral_mismatch=boson_sampler_params.get("spectral_mismatch", 0.0),
                    arrival_time_jitter=boson_sampler_params.get("arrival_time_jitter", 0.0),
                    bs_loss=boson_sampler_params.get("bs_loss", 0.0),
                    bs_jitter=boson_sampler_params.get("bs_jitter", 0.0),
                    phase_noise_std=boson_sampler_params.get("phase_noise_std", 0.0),
                    systematic_phase_offset=boson_sampler_params.get("systematic_phase_offset", 0.0),
                    mode_loss=boson_sampler_params.get("mode_loss"),
                    dark_count_rate=boson_sampler_params.get("dark_count_rate", 0.0),
                    use_advanced_nonlinearity=boson_sampler_params.get("use_advanced_nonlinearity", False),
                    detuning=boson_sampler_params.get("detuning", 0.0),
                    pulse_bw=boson_sampler_params.get("pulse_bw", 0.0),
                    QD_linewidth=boson_sampler_params.get("QD_linewidth", 0.0),
                    phi=boson_sampler_params.get("phi", 0.0),
                    g2_target=boson_sampler_params.get("g2_target", None),
                )
                self.boson_sampler = BosonLatentGenerator(self.latent_features, bs_module)

        self.register_buffer(
            'prior_params',
            torch.zeros(1, 2 * latent_features)
        )

    def posterior(self, x: torch.Tensor) -> Distribution:
        h_x = self.encoder(x)
        mu, log_sigma = h_x.chunk(2, dim=-1)
        return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def prior(self, batch_size: int) -> Distribution:
        if self.boson_sampler is not None:
            return BosonPrior(
                boson_sampler=self.boson_sampler,
                batch_size=batch_size,
                latent_features=self.latent_features
            )
        else:
            prior_params = self.prior_params.expand(batch_size, -1)
            mu, log_sigma = prior_params.chunk(2, dim=-1)
            return ReparameterizedDiagonalGaussian(mu, log_sigma)

    def observation_model(self, z: torch.Tensor) -> Distribution:
        # decode to a real-valued “mean” for each pixel
        px_loc = self.decoder(z).view(-1, *self.input_shape)
        # choose a fixed or learnable scale (here we use 1.0)
        px_scale = torch.ones_like(px_loc)  
        return Normal(loc=px_loc, scale=px_scale)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        qz = self.posterior(x)
        pz = self.prior(x.size(0))
        z = qz.rsample()
        px = self.observation_model(z)
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}

class VariationalInference(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta

    def forward(self, model, x, target=None):
        outputs = model(x)
        px, pz, qz, z = outputs['px'], outputs['pz'], outputs['qz'], outputs['z']

        def reduce_mean(tensor):
            return tensor.view(tensor.size(0), -1).mean(dim=1)

        if target is None:
            x_target = x.view(x.size(0), *model.input_shape)
        else:
            x_target = target

        log_px = reduce_mean(px.log_prob(x_target))
        log_pz = reduce_mean(pz.log_prob(z))
        log_qz = reduce_mean(qz.log_prob(z))

        kl = log_qz - log_pz
        beta_elbo = log_px - self.beta * kl
        loss = -beta_elbo.mean()

        diagnostics = {
            'elbo': (log_px - kl).detach(),
            'log_px': log_px.detach(),
            'kl': kl.detach()
        }
        return loss, diagnostics, outputs
