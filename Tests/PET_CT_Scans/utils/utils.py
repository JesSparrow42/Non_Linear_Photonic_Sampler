import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def save_weights(model, optimizer, epoch, save_path, loss):
    """Save the model weights and optimizer state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_path)

def load_weights(model, optimizer, load_path, device):
    """Load the model weights and optimizer state."""
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', None)
        print(f"Loaded model from {load_path}, starting from epoch {epoch}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {load_path}, starting from scratch.")
        return 0, None
    
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
    plt.show()
    plt.savefig('training.png')

def calculate_final_layer_params(output_size, initial_size=(256,256), stride=2):
    """
    Calculate kernel size, stride, and padding for the final layer of the generator.
    Ensures compatibility with target output size.
    """
    target_h, target_w = output_size
    initial_h, initial_w = initial_size

    # Calculate kernel size to achieve target size
    kernel_size_h = target_h - (initial_h - 1) * stride
    kernel_size_w = target_w - (initial_w - 1) * stride

    # Ensure kernel size is valid
    kernel_size_h = max(1, kernel_size_h)
    kernel_size_w = max(1, kernel_size_w)

    # Compute padding to maintain dimensions
    padding_h = max(0, (kernel_size_h - 1) // 2)
    padding_w = max(0, (kernel_size_w - 1) // 2)

    return (kernel_size_h, kernel_size_w), stride, (padding_h, padding_w)

