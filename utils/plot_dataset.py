from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from data import SpringDataset
from utils.plot import get_spectrogram


def visualize_data(**args):
    

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    ax.plot(x, alpha=0.7, label='Ground Truth', color='blue')
    ax.plot(y, alpha=0.7, label='Prediction', color='red')

    ax.set_title('Waveform')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(args.results_dir) / 'dataset_wave.png')

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    # Plot input spectrogram
    axs[0].imshow(10 * np.log10(x), aspect='auto', origin='lower')
    axs[0].set_title('Input Spectrogram')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    axs[0].colorbar(label='dB')

    # Plot target spectrogram
    axs[1].imshow(10 * np.log10(y), aspect='auto', origin='lower')
    axs[1].set_title('Target Spectrogram')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    axs[1].colorbar(label='dB')

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(args.results_dir) / 'dataset_spectrum.png')

if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    
    dataset = SpringDataset(args.data_dir, args.split)
    
    x, y = dataset[args.sample_idx]
    visualize_data(x, y, **args)

    print("Plotting requested samples...")
