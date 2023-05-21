import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from .. import dataload
from sandbox.config import *

def visualize_and_play(x, y, fs=SAMPLE_RATE):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    # Display the waveform
    ax[0].plot(x)
    ax[0].set_title('Waveform')
    ax[0].set_xlabel('Sample Index')
    ax[0].set_ylabel('Amplitude')

    # Compute and display the spectrogram
    f, t, Sxx = spectrogram(y, fs)
    ax[1].pcolormesh(t, f, 10 * np.log10(Sxx))
    ax[1].set_title('Spectrogram')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Frequency [Hz]')

    plt.tight_layout()

# Load the training and test datasets
train_dataset = dataload.PlateSpringDataset(DATA_DIR, split='train')
test_dataset = dataload.PlateSpringDataset(DATA_DIR, split='test')

# Visualize and play a sample from each split
print("Training sample:")
x_train, y_train = train_dataset[0]
visualize_and_play(x_train, y_train)

print("Test sample:")
x_test, y_test = test_dataset[0]
visualize_and_play(x_test, y_test)

plt.show()

samples_list = [1, 5, 6, 7, 11]

# Create a figure with subplots
fig, axs = plt.subplots(nrows=1, ncols=len(samples_list), figsize=(20, 4))

# Loop through the indices and plot the waveform for each example
for i, sample_idx in enumerate(samples_list):
    x, y = train_dataset[sample_idx]
    axs[i].plot(x)
    axs[i].plot(y)
    axs[i].set_title(f'Sample {sample_idx}')
    axs[i].set_xlabel('Time (samples)')
    axs[i].set_ylabel('Amplitude')
    axs[i].legend(['Input', 'Target'])

plt.show()
