import torch
import torch.nn.functional as F
from tests.PET_CT_Scans.gan_module import Generator, Discriminator
from ptseries.models import PTGenerator
from tests.PET_CT_Scans.utils.data_loader import create_data_loader
import os
from pathlib import Path
from PIL import Image
import numpy as np
from bosonsampler import BosonSamplerTorch, BosonLatentGenerator

def denormalize(tensor):
    """
    Convert a tensor from the [-1, 1] range to [0, 1].
    """
    return (tensor + 1) / 2.0

def load_generator(model_path, device, input_state_length):
    # Initialize the generator model.
    # Note: In train.py, Generator is instantiated with a single argument (latent_dim).
    generator = Generator(input_state_length)
    generator.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    generator.to(device)
    generator.eval()  # Set the generator to evaluation mode
    return generator

def generate_ct_images(generator, latent_generator, data_loader, device, output_folder='generated_ct_images'):
    """
    Generate CT images using the provided generator and latent generator.

    The latent_generator can either be:
      - An object with a .generate(num_samples) method (like PTGenerator), or
      - A callable (like BosonLatentGenerator, which defines forward() and hence __call__).

    This function automatically checks which method is available.
    """
    os.makedirs(output_folder, exist_ok=True)
    for i, (pet_images, _) in enumerate(data_loader):
        pet_images = pet_images.to(device)
        batch_size = pet_images.size(0)
        # Get latent vectors from the latent generator.
        if hasattr(latent_generator, "generate"):
            latent_vectors = latent_generator.generate(batch_size)
        else:
            latent_vectors = latent_generator(batch_size)
        latent_vectors = latent_vectors.to(device)
        
        # Generate CT images from the latent vectors.
        with torch.no_grad():
            generated_ct_images = generator(latent_vectors)
        # Denormalize images from [-1,1] to [0,1].
        generated_ct_images = denormalize(generated_ct_images)

        # Save each generated image.
        for j in range(batch_size):
            image_path = os.path.join(output_folder, f'generated_ct_image_{i * batch_size + j}.png')
            save_image(generated_ct_images[j], image_path)
            print(f"Saved generated CT image at {image_path}")

def save_image(tensor, path):
    """
    Convert a tensor image to a NumPy array, scale it to [0, 255],
    and save it using PIL.
    """
    # Convert the tensor to a NumPy array.
    image = tensor.cpu().numpy()
    # Check if the image is grayscale (single channel) or RGB (3 channels).
    if image.shape[0] == 1:  # Grayscale image.
        image = image.squeeze(0)
    else:
        image = image.transpose(1, 2, 0)  # Convert from CHW to HWC.
    image = image * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)

def main():
    # Hyperparameters (as in your training script)
    boson_sampler_params = {
        "input_state": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
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
    
    # Paths (adjust these paths as needed)
    ct_folder = '/Users/olivernorregaard/Downloads/DiscretePETCT-main/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/2.000000-CT IMAGES-25805'
    pet_folder = '/Users/olivernorregaard/Downloads/DiscretePETCT-main/NAC-PET & CT/ACRIN-NSCLC-FDG-PET/ACRIN-NSCLC-FDG-PET-016/12-24-1959-NA-NA-02783/1.000000-PET NAC-24000'
    output_folder = 'tests/PET_CT_Scans/generated_ct_images'
    
    # Set device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Load the generator model.
    input_state_length = len(boson_sampler_params["input_state"])
    
    # Paths to generator model checkpoints for each latent generator
    generator_model_paths = {
        "gaussian":        "model_checkpoints/gaussian_generator_epoch_1500.pt",
        "ptseries":        "model_checkpoints/pt_generator_epoch_1500.pt",
        "boson_linear":    "model_checkpoints/boson_generator_epoch_1500.pt",
        "boson_nonlinear": "model_checkpoints/boson_nonlinear_generator_epoch_1500.pt"
    }
    
    # Create latent generators.
    # Option 1: PTSeries latent generator.
    pt_latent_space = PTGenerator(**boson_sampler_params)
    
    # Option 2: Boson-based latent generator.
    latent_dim = len(boson_sampler_params["input_state"])
    boson_nonlinear_torch = BosonSamplerTorch(
        m=10,
        num_sources=5,
        num_loops=100,
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
        mode_loss=np.ones(10),
        dark_count_rate=0,
        use_advanced_nonlinearity=False,
        g2_target=1
    )
    boson_nonlinear_latent_space = BosonLatentGenerator(latent_dim, boson_nonlinear_torch)

    # Boson linear latent generator (no nonlinearity)
    boson_linear_torch = BosonSamplerTorch(
        m=10,
        num_sources=5,
        num_loops=100,
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
        mode_loss=np.ones(10),
        dark_count_rate=0,
        use_advanced_nonlinearity=True,
        g2_target=1
    )
    boson_linear_latent_space = BosonLatentGenerator(latent_dim, boson_linear_torch)

    # Gaussian latent generator
    def gaussian_latent_space(batch_size):
        return torch.randn(batch_size, input_state_length, device=device)

    # Collect all latent generators
    latent_generators = {
        "gaussian": gaussian_latent_space,
        "ptseries": pt_latent_space,
        "boson_linear": boson_linear_latent_space,
        "boson_nonlinear": boson_nonlinear_latent_space
    }
    
    # Create a data loader for PET images only (CT images not needed for latent generation).
    pet_data_loader = create_data_loader(pet_folder=pet_folder, ct_folder=ct_folder, num_workers=4, shuffle=False)

    for name, latent_space in latent_generators.items():
        # Load the appropriate generator model for this latent generator
        model_path = generator_model_paths[name]
        generator = load_generator(model_path, device, input_state_length)
        series_output = os.path.join(output_folder, name)
        print(f"Generating CT images with {name} latent generator into {series_output}...")
        generate_ct_images(generator, latent_space, pet_data_loader, device, series_output)

if __name__ == '__main__':
    main()
