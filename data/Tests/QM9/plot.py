#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_with_std(group, metrics, ylabel, title):
    plt.figure(figsize=(8, 5))
    for col, label in metrics:
        stats  = group[col].agg(['mean', 'std'])
        epochs = stats.index.values
        mean   = stats['mean'].values
        std    = stats['std'].values
        plt.plot(epochs, mean, label=label)
        plt.fill_between(epochs, mean - std, mean + std, alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    csv_path   = '/Users/olivernorregaard/Documents/GitHub/nonlinear-photonic-sampler/resultsNEW.csv'

    # load
    df    = pd.read_csv(csv_path, engine='python')
    group = df.groupby('epoch')

    # define metrics
    discrim = [
        ('d_loss_base', 'Gaussian'),
        ('d_loss_bos',  'Boson'),
        #('d_loss_nl',   'Nonlinear'),
    ]
    gen     = [
        ('g_loss_base', 'Gaussian'),
        ('g_loss_bos',  'Boson'),
        #('g_loss_nl',   'Nonlinear'),
    ]
    w1      = [
        ('w1_base',    'Gaussian'),
        ('w1_bos',     'Boson'),
        #('w1_nl',      'Nonlinear'),
    ]
    time_ms = [
        ('baseline_train_ms',    'Gaussian'),
        ('boson_train_ms',       'Boson'),
        #('boson_nl_train_ms',    'Nonlinear'),
    ]

    # plots
    plot_with_std(group, discrim, 'Discriminator Loss', 'Discriminator Loss vs. Epoch')
    plot_with_std(group, gen,     'Generator Loss',     'Generator Loss vs. Epoch')
    plot_with_std(group, w1,      'W1 Distance',        'W1 Distance vs. Epoch')
    plot_with_std(group, time_ms, 'Training Time (ms)', 'Training Time vs. Epoch')


    # 5) Generator training speed vs. Epoch (negative derivative of generator loss, smoothed)
    plt.figure(figsize=(8, 5))
    for col, label in gen:
        # compute mean loss per epoch
        loss_series = group[col].mean()
        # negative epoch‐to‐epoch derivative
        speed_series = -loss_series.diff().dropna()
        # apply centered rolling average (3‐epoch window)
        speed_smooth = speed_series.rolling(window=3,
                                            min_periods=1,
                                            center=True).mean()
        plt.plot(speed_smooth.index, speed_smooth.values, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Training Speed (-ΔGenerator Loss)')
    #plt.title('Generator Training Speed vs. Epoch (smoothed)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 6) Reward vs. Epoch
    reward_metrics = [
        ('reward_base', 'Gaussian'),
        ('reward_bos',  'Boson'),
        #('reward_nl',   'Nonlinear'),
    ]
    plot_with_std(group, reward_metrics, 'Reward', 'Reward vs. Epoch')

if __name__ == '__main__':
    main()