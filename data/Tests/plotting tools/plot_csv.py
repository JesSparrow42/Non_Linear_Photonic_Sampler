import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Load the data
def plot_qgan_losses(csv_file='training_data/mnist_test.csv'):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Group by Run to handle multiple training runs
    runs = df['Run'].unique()
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 0.5]})
    
    # Iterate through runs
    colors = ["b", "r", "o"]
    
    for i, run in enumerate(runs):
        run_data = df[df['Run'] == run]
        
        # Sort by epoch for proper plotting
        run_data = run_data.sort_values('Epoch')
        
        # First subplot: All losses on the same graph
        ax1 = axes[0]
        ax1.plot(run_data['Epoch'], run_data['PT_Gen_Loss'], 
                marker='o', linestyle='-', color='blue', alpha=0.7, 
                label=f'Run {i+1}: PT Generator')
        ax1.plot(run_data['Epoch'], run_data['Boson_Gen_Loss'], 
                marker='s', linestyle='--', color='red', alpha=0.7, 
                label=f'Run {i+1}: Boson Generator')
        ax1.plot(run_data['Epoch'], run_data['Boson_Gen_NL_Loss'], 
                marker='^', linestyle=':', color='orange', alpha=0.7, 
                label=f'Run {i+1}: Boson NL Generator')
    
    # Customize first subplot
    ax1.set_title('Generator Loss Comparison Across Training Runs', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10, loc='upper right')
    
    # Second subplot: Difference between models
    for i, run in enumerate(runs):
        run_data = df[df['Run'] == run]
        run_data = run_data.sort_values('Epoch')
        
        # Calculate differences
        boson_pt_diff = run_data['Boson_Gen_Loss'] - run_data['PT_Gen_Loss']
        boson_nl_pt_diff = run_data['Boson_Gen_NL_Loss'] - run_data['PT_Gen_Loss']
        
        # Plot differences
        ax2 = axes[1]
        ax2.plot(run_data['Epoch'], boson_pt_diff, 
                marker='s', linestyle='--', color='red', alpha=0.7,
                label=f'Run {i+1}: Boson - PT')
        ax2.plot(run_data['Epoch'], boson_nl_pt_diff, 
                marker='^', linestyle=':', color='orange', alpha=0.7,
                label=f'Run {i+1}: Boson NL - PT')
        
        # Add zero line
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Customize second subplot
    ax2.set_title('Difference in Loss (Relative to PT Generator)', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss Difference', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10, loc='upper right')
    
    # Additional plot showing loss convergence rate
    fig2, ax3 = plt.subplots(figsize=(12, 6))
    
    for i, run in enumerate(runs):
        run_data = df[df['Run'] == run]
        run_data = run_data.sort_values('Epoch')
        
        # Normalize losses to starting value for comparison
        pt_norm = run_data['PT_Gen_Loss'] / run_data['PT_Gen_Loss'].iloc[0]
        boson_norm = run_data['Boson_Gen_Loss'] / run_data['Boson_Gen_Loss'].iloc[0]
        boson_nl_norm = run_data['Boson_Gen_NL_Loss'] / run_data['Boson_Gen_NL_Loss'].iloc[0]
        
        # Plot normalized losses
        ax3.plot(run_data['Epoch'], pt_norm, 
                marker='o', linestyle='-', color='blue', alpha=0.7,
                label=f'Run {i+1}: PT Generator (normalized)')
        ax3.plot(run_data['Epoch'], boson_norm, 
                marker='s', linestyle='--', color='red', alpha=0.7,
                label=f'Run {i+1}: Boson Generator (normalized)')
        ax3.plot(run_data['Epoch'], boson_nl_norm, 
                marker='^', linestyle=':', color='orange', alpha=0.7,
                label=f'Run {i+1}: Boson NL Generator (normalized)')
    
    # Customize convergence plot
    ax3.set_title('Convergence Rate Comparison (Normalized to Starting Loss)', fontsize=16)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Normalized Loss', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=10, loc='upper right')
    
    # Make plots look nice
    plt.tight_layout()
    
    # Save the figures
    fig.savefig('qgan_loss_comparison.png', dpi=300, bbox_inches='tight')
    fig2.savefig('qgan_convergence_rate.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
if __name__ == "__main__":
    plot_qgan_losses(csv_file='training_data/mnist_test_updated_Kerr.csv')