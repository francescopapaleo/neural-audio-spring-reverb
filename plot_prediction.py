from config import *
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

import torchaudio.transforms as T
import torch


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

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
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

    fig, ax = plt.subplots(nrows=1, ncols=1)

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


def plot_compare_spectrogram(spec1, spec2, spec3, titles=['Title1', 'Title2', 'Title3'], ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5)) # 1 row, 3 columns

    for idx, spec in enumerate([spec1, spec2, spec3]):
        axs[idx].set_title(titles[idx])
        axs[idx].set_ylabel(ylabel)
        axs[idx].set_xlabel("frame")
        im = axs[idx].imshow(torch.log10(spec[0]), origin="lower", aspect=aspect, cmap='hot')
        if xmax:
            axs[idx].set_xlim((0, xmax))
        fig.colorbar(im, ax=axs[idx])

    # Ensure RESULTS directory exists
    Path(RESULTS).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(Path(RESULTS) / 'spectrogram.png')
    plt.show()
