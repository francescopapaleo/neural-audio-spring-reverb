from config import *
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def visualize_samples(x, y, fs=sample_rate):
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
    plt.savefig


def plot_compare_waveform(y, y_pred, fs=sample_rate):
    '''Plot the waveform of the input, the prediction and the ground truth
    Parameters
    ----------
    y : array_like
        Ground truth signal
    y_pred : array_like
        The predicted signal
    fs : int, optional
        The sampling frequency (default to 1, i.e., samples).'''

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.plot(y, alpha=0.7, label='Ground Truth', color='blue')
    ax.plot(y_pred, alpha=0.7, label='Prediction', color='red')

    ax.set_title('Waveform')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()

    # Save the figure
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(RESULTS) / 'waveform_plot.png')
    plt.show()

def plot_zoom_waveform(y, y_pred, fs=sample_rate, t_start=None, t_end=None):
    '''Plot the waveform of the ground truth and the prediction
    Parameters
    ----------
    y : array_like
        Ground truth signal
    y_pred : array_like
        The predicted signal
    fs : int, optional
        The sampling frequency (default to 1, i.e., samples).
    t_start : float, optional
        The start time of the plot (default to None).
    t_end : float, optional
        The end time of the plot (default to None).'''
    
    # Create a time array
    t = np.arange(y.shape[0]) / fs

    # Determine the indices corresponding to the start and end times
    if t_start is not None:
        i_start = int(t_start * fs)
    else:
        i_start = 0

    if t_end is not None:
        i_end = int(t_end * fs)
    else:
        i_end = len(t)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    ax.plot(t[i_start:i_end], y[i_start:i_end], alpha=0.7, label='Ground Truth', color='blue')
    ax.plot(t[i_start:i_end], y_pred[i_start:i_end], alpha=0.7, label='Prediction', color='red')

    ax.set_title('Waveform')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()

    # Ensure RESULTS directory exists
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(RESULTS) / 'waveform_zoom.png')

    plt.show()


def plot_compare_spectrum(y, y_pred, x, fs=sample_rate):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    epsilon = 1e-10  # add a small constant to avoid log(0)

    axs[0].specgram(y, NFFT=2048, Fs=fs, noverlap=1024, cmap='hot')
    axs[0].set_title('Spectrogram y')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Frequency [Hz]')
    
    axs[1].specgram(y_pred, NFFT=2048, Fs=fs, noverlap=1024, cmap='hot')
    axs[1].set_title('Spectrogram y_pred')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Frequency [Hz]')
    
    axs[2].specgram(x+epsilon, NFFT=2048, Fs=fs, noverlap=1024, cmap='hot')
    axs[2].set_title('Spectrogram x')
    axs[2].set_xlabel('Time [s]')
    axs[2].set_ylabel('Frequency [Hz]')

    plt.tight_layout()
     # Ensure RESULTS directory exists
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(RESULTS) / 'spectrogram.png')

    plt.show()

import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torch

def get_spectrogram(
    waveform,
    n_fft=400,
    win_len=None,
    hop_len=None,
    power=2.0,
):
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )
    return spectrogram(waveform)

def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(torch.log10(spec[0]), origin="lower", aspect=aspect, cmap='hot')
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

