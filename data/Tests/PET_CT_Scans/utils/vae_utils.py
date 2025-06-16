# utils.py
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.distributions import Distribution

class BosonPrior(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, boson_sampler, batch_size, latent_features, discriminator_iter=0, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.boson_sampler = boson_sampler
        self.batch_size = batch_size
        self.latent_features = latent_features
        self.discriminator_iter = discriminator_iter

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    def rsample(self, sample_shape=torch.Size()):
        total_samples = (1 + self.discriminator_iter) * self.batch_size
        latent = self.boson_sampler.generate(total_samples).to(self.device)
        latent = torch.chunk(latent, 1 + self.discriminator_iter, dim=0)
        return latent

    def log_prob(self, z):
        mean = torch.zeros_like(z)
        std = torch.ones_like(z)
        log_scale = torch.log(std)
        log_prob = -0.5 * (
            ((z - mean) ** 2) / (std ** 2)
            + 2 * log_scale
            + torch.log(torch.tensor(2 * torch.pi, device=z.device))
        )
        return log_prob.sum(dim=-1)

class ReparameterizedDiagonalGaussian(Distribution):
    arg_constraints = {}
    has_rsample = True

    def __init__(self, mu: torch.Tensor, log_sigma: torch.Tensor, validate_args=None):
        super().__init__(validate_args=validate_args)
        self.mu = mu
        self.sigma = log_sigma.exp()

    def sample_epsilon(self) -> torch.Tensor:
        return torch.empty_like(self.mu).normal_()

    def sample(self) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample()

    def rsample(self) -> torch.Tensor:
        epsilon = self.sample_epsilon()
        return self.mu + self.sigma * epsilon

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        log_scale = torch.log(self.sigma)
        return -0.5 * (
            ((z - self.mu) ** 2) / (self.sigma ** 2)
            + 2 * log_scale
            + torch.log(torch.tensor(2 * torch.pi, device=z.device))
        )

def save_images(original, reconstructed, output_dir_orig, output_dir_reco, epoch, expected_shape=None):
    """Save original and reconstructed images.
    If expected_shape is provided and an image is flat (1D), it will be reshaped.
    Otherwise, the image is saved as-is.
    """
    os.makedirs(output_dir_orig, exist_ok=True)
    os.makedirs(output_dir_reco, exist_ok=True)
    for i, img in enumerate(original):
        if expected_shape is not None and img.ndim == 1:
            try:
                img = img.reshape(expected_shape)
            except Exception as e:
                print(f"Error reshaping original image {i}: {e}")
        plt.imsave(os.path.join(output_dir_orig, f"epoch_{epoch}_original_{i}.png"), img, cmap="gray")
    for i, img in enumerate(reconstructed):
        if expected_shape is not None and img.ndim == 1:
            try:
                img = img.reshape(expected_shape)
            except Exception as e:
                print(f"Error reshaping reconstructed image {i}: {e}")
        plt.imsave(os.path.join(output_dir_reco, f"epoch_{epoch}_reconstructed_{i}.png"), img, cmap="gray")


def save_weights(model, optimizer, epoch, save_path, loss):
    """Save the model weights and optimizer state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)


def load_separate_checkpoints(self, gen_checkpoint_path, disc_checkpoint_path, device):
        # Load generator checkpoint
        if os.path.exists(gen_checkpoint_path):
            gen_ckpt = torch.load(gen_checkpoint_path, map_location=device)
            self.generator.load_state_dict(gen_ckpt['model_state_dict'])
            start_epoch_gen = gen_ckpt.get('epoch', 0)
            print(f"Loaded generator checkpoint from {gen_checkpoint_path} at epoch {start_epoch_gen}")
        else:
            start_epoch_gen = 0
            print("No generator checkpoint found; starting generator from scratch.")

        # Load discriminator checkpoint
        if os.path.exists(disc_checkpoint_path):
            disc_ckpt = torch.load(disc_checkpoint_path, map_location=device)
            self.discriminator.load_state_dict(disc_ckpt['model_state_dict'])
            start_epoch_disc = disc_ckpt.get('epoch', 0)
            print(f"Loaded discriminator checkpoint from {disc_checkpoint_path} at epoch {start_epoch_disc}")
        else:
            start_epoch_disc = 0
            print("No discriminator checkpoint found; starting discriminator from scratch.")

        return start_epoch_gen, start_epoch_disc


def plot_losses(gen_losses, disc_losses, phase):
    """Plot generator and discriminator losses."""
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{phase} Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('training.png')
    plt.show()

def dice_loss(pred, target):
    """Compute the dice loss between prediction and target."""
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - ((2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth))

def generator_loss(disc_output, real_ct, generated_ct, bone_mask=None):
    """Compute a combined generator loss."""
    if bone_mask is None:
        L_disc = F.binary_cross_entropy(disc_output, torch.ones_like(disc_output))
        L_MAE = F.l1_loss(generated_ct, real_ct)
        L_dice = dice_loss(generated_ct, real_ct)
    else:
        L_disc = F.binary_cross_entropy(disc_output, torch.ones_like(disc_output))
        L_MAE = F.l1_loss(generated_ct, real_ct)
        L_dice = dice_loss(generated_ct * bone_mask, real_ct * bone_mask)
    return L_disc + 150 * L_MAE + L_dice

def plot_molecule(node_positions, node_features, save_path):
    """
    Plot a molecule given its node positions and features, and save the plot.
    """
    try:
        plt.figure()
        # node_positions and node_features are already numpy arrays.
        # node_positions.shape -> (num_atoms, 3) or (num_atoms, 2)
        # node_features.shape -> (num_atoms,) or (num_atoms, 1)

        x = node_positions[:, 0]
        y = node_positions[:, 1]

        # Convert node_features to 1D if it has shape (num_atoms, 1)
        if node_features.ndim == 2:
            node_features = node_features.squeeze(axis=-1)

        # Simple 2D scatter. You can color-code by atomic number or any feature.
        sc = plt.scatter(x, y, c=node_features, cmap='viridis', s=100, edgecolors='k')
        plt.colorbar(sc, label='Node feature')

        plt.title("Molecule")
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.savefig(save_path)
        plt.close()
        print(f"Molecule successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving molecule plot: {e}")
