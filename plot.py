from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('MacOSX')  # Replace 'TkAgg' with an alternative backend

from scipy.signal import spectrogram

import torchaudio.transforms as T
import torch

from config import parser

args = parser.parse_args()
sample_rate = args.sr
results_dir = Path(args.results_dir)

def plot_compare_waveform(y, y_pred):
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
    plt.savefig(Path(args.results_dir) / 'waveform_plot.png')
    plt.close(fig)
    print("Saved waveform plot to: ", Path(args.results_dir) / 'waveform_plot.png')

def plot_zoom_waveform(y, y_pred, sample_rate, t_start=None, t_end=None):
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
    t = np.arange(y.shape[0]) / sample_rate

    # Determine the indices corresponding to the start and end times
    if t_start is not None:
        i_start = int(t_start * sample_rate)
    else:
        i_start = 0

    if t_end is not None:
        i_end = int(t_end * sample_rate)
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

    plt.savefig(Path(args.results_dir) / 'waveform_zoom.png')
    plt.close(fig)
    print("Saved zoomed waveform plot to: ", Path(args.results_dir) / 'waveform_zoom.png')

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
    plt.tight_layout()
    plt.savefig(Path(args.results_dir) / 'spectrogram.png')
    plt.close(fig)
    print("Saved spectrogram plot to: ", Path(args.results_dir) / 'spectrogram.png')

def plot_signals(sweep_filt, inverse_filter, measured, SAMPLE_RATE, duration, file_name):
    time_stamps = np.arange(0, duration, 1/ args.sample_rate)
    fig, ax = plt.subplots(3, 1, figsize=(15,7))
    
    ax[0].plot(time_stamps, sweep_filt)
    ax[0].set_xlim([0, time_stamps[-1]])
    ax[0].set_title("Sweep Tone")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Amplitude")
    
    ax[1].plot(time_stamps, inverse_filter)
    ax[1].set_xlim([0, time_stamps[-1]])
    ax[1].set_title("Inverse Filter")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Amplitude")
    
    time_stamps = np.arange(0, len(measured)/ sample_rate, 1/ sample_rate)
    ax[2].plot(time_stamps, measured)
    ax[2].set_xlim([0, time_stamps[-1]])
    ax[2].set_title("Impulse Response")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(Path(args.results_dir) / file_name)
    plt.close(fig)
    print("Saved signal plot to: ", Path(args.results_dir) / file_name)

def plot_transfer_function(magnitude, phase, sample_rate, file_name):
    freqs = np.linspace(0, sample_rate / 2, len(magnitude))
    
    fig, ax = plt.subplots(2, 1, figsize=(15, 7))
    ax[0].semilogx(freqs, magnitude)
    ax[0].set_xlim([1, freqs[-1]])
    ax[0].set_ylim([-100, 6])
    ax[0].set_xlabel("Frequency [Hz]")
    ax[0].set_ylabel("Magnitude [dBFS]")
    ax[1].semilogx(freqs, phase)
    ax[1].set_xlim([1, freqs[-1]])
    ax[1].set_ylim([-180, 180])
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Phase [degrees]")
    plt.suptitle("H(w) - Transfer Function")
    plt.tight_layout()
    plt.savefig(Path(args.results_dir) / file_name)
    plt.close(fig)
    print("Saved transfer function plot to: ", Path(args.results_dir) / file_name)
