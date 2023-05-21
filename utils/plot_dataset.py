import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from dataload import PlateSpringDataset
from config import *
from pathlib import Path
from argparse import ArgumentParser

# Create the argument parser
parser = ArgumentParser()
parser.add_argument('--sample_idx', type=int, default=0, help='The index of the sample to visualize')
args = parser.parse_args()


def visualize_data(x, y, fs=SAMPLE_RATE):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    ax.plot(x, alpha=0.7, label='Ground Truth', color='blue')
    ax.plot(y, alpha=0.7, label='Prediction', color='red')

    ax.set_title('Waveform')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(RESULTS) / 'dataset_wave.png')

    # Create a figure with subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))

    # Plot input spectrogram
    axs[0].imshow(10 * np.log10(x_spec), aspect='auto', origin='lower')
    axs[0].set_title('Input Spectrogram')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    axs[0].colorbar(label='dB')

    # Plot target spectrogram
    axs[1].imshow(10 * np.log10(y_spec), aspect='auto', origin='lower')
    axs[1].set_title('Target Spectrogram')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    axs[1].colorbar(label='dB')

    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(RESULTS) / 'dataset_spectrum.png')

if __name__ == "__main__":
    
    train_dataset = PlateSpringDataset(DATA_DIR, split='train')
    x_train, y_train = train_dataset[args.sample_idx]
    visualize_data(x_train, y_train)

    print("Plotting requested samples...")
