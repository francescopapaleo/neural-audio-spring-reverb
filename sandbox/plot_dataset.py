import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_load import SubsetRetriever
from config import *

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import spectrogram

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

# Load the data
subset_retriever = SubsetRetriever(DATA_DIR)
x_train, y_train, x_test, y_test = subset_retriever.retrieve_data(concatenate=True)

# Visualize and play a sample from each split
print("Training sample:")
visualize_and_play(x_train[0], y_train[0])
print("Test sample:")
visualize_and_play(x_test[0], y_test[0])

plt.show()


samples_list = [1, 5, 6, 7, 11]

# Create a figure with subplots
fig, axs = plt.subplots(nrows=1, ncols=len(samples_list), figsize=(20, 4))

# Loop through the indices and plot the waveform for each example
for i, sample_idx in enumerate(samples_list):
    x = Xtrain[sample_idx, :, 0]
    y = Ytrain_0[sample_idx, :, 0]
    axs[i].plot(x)
    axs[i].plot(y)
    axs[i].set_title(f'Sample {sample_idx}')
    axs[i].set_xlabel('Time (samples)')
    axs[i].set_ylabel('Amplitude')
    axs[i].legend(['Input', 'Target'])

plt.show()
