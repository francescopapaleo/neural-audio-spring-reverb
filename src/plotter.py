"""" Plotting utilities for the project.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from pathlib import Path
from configurations import parse_args
from scipy.signal import spectrogram

def save_plot(plt, file_name):
    args = parse_args()
    plot_dir = Path(args.plotsdir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    file_path = plot_dir / (file_name + ".png")
    plt.tight_layout()
    plt.savefig(file_path)
    print(f"Saved plot to {file_path}")


def apply_decorations(ax, legend=False, location="upper right"):
    ax.grid(True)
    if legend:
        ax.legend()
        ax.legend(loc=location)


def plot_data(x, y, subplot, title, x_label, y_label, legend=False):
    min_length = min(len(x), len(y))
    x = x[:min_length]
    y = y[:min_length]
    subplot.plot(x, y)
    subplot.set_xlim([0, x[-1]])
    subplot.set_title(title)
    subplot.set_xlabel(x_label)
    subplot.set_ylabel(y_label)

    if x_label == "Frequency [Hz]":
        subplot.set_xscale("symlog")
        subplot.set_xlim([20, 12000])
        formatter = ticker.ScalarFormatter()
        formatter.set_scientific(False)
        subplot.xaxis.set_major_formatter(formatter)
        subplot.xaxis.set_minor_formatter(formatter)
        
    apply_decorations(subplot, legend)


def get_time_stamps_np(signal_length, sample_rate):
    return np.linspace(0, (signal_length-1) / sample_rate, signal_length)


def plot_impulse_response(sweep: np.ndarray, inverse_filter: np.ndarray, measured: np.ndarray, sample_rate: int, file_name: str):
    fig, ax = plt.subplots(3, 1, figsize=(15,7))
    plot_data(get_time_stamps_np(len(sweep), sample_rate), sweep, ax[0], "Processed Sweep Tone", "Time [s]", "Amplitude")
    plot_data(get_time_stamps_np(len(inverse_filter), sample_rate), inverse_filter, ax[1], "Inverse Filter", "Time [s]", "Amplitude")
    plot_data(get_time_stamps_np(len(measured), sample_rate), measured, ax[2], "Impulse Response", "Time [s]", "Amplitude")
    fig.suptitle(f"{file_name} - Impulse Response δ(t)")
    save_plot(fig, file_name + "_IR")

def generate_spectrogram(waveform, sample_rate):
    frequencies, times, Sxx = spectrogram(
        waveform, 
        fs=sample_rate, 
        window='blackmanharris',
        nperseg=64,
        noverlap=32,
        # nfft=2048,  
        scaling='density', 
        mode='magnitude'
    )
    
    # Convert magnitude to dB
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    
    return frequencies, times, Sxx_dB


def plot_waterfall(waveform, file_name, sample_rate, stride=1):
    frequencies, times, Sxx = generate_spectrogram(waveform, sample_rate)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    # Compute logarithm of the frequencies (avoid log(0) by adding a small value)
    # log_freq = np.log10(frequencies + 1e-10)
    
    # X, Y = np.meshgrid(frequencies, times[::stride])
    X, Y = np.meshgrid(frequencies, times[::stride])
    Z = Sxx.T[::stride]

    surf = ax.plot_surface(X, Y, Z, 
                           cmap='inferno',
                           edgecolor='none', 
                           alpha=0.8,
                           linewidth=0,
                            antialiased=False)

    # Add a color bar which maps values to colors
    cbar = fig.colorbar(surf, ax=ax, pad=0.01, aspect=35, shrink=0.5)
    cbar.set_label('Magnitude (dB)')

    # Set labels and title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time (seconds)')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title(f'{file_name} - Spectrogram')
    
    ax.set_xlim([frequencies[-1], frequencies[0]])
    ax.view_init(15, 15)  # Adjusts the viewing angle for better visualization
    plt.tight_layout()

    plt.tight_layout()
    plt.savefig(f"results/measured_IR/{file_name}_waterfall.png", dpi=300)
    print(f"Saved spectrogram plot to {file_name}_waterfall.png")


def plot_rt60(T, energy_db, e_5db, est_rt60, rt60_tgt, file_name):
    plt.subplots(figsize=(6, 6))
    plt.plot(T, energy_db, label="Energy")
    plt.plot([0, est_rt60], [e_5db, -65], "--", label="Linear Fit")
    plt.plot(T, np.ones_like(T) * -60, "--", label="-60 dB")
    plt.vlines(
        est_rt60, energy_db[-1], 0, linestyles="dashed", label="Estimated RT60"
    )

    if rt60_tgt is not None:
        plt.vlines(rt60_tgt, energy_db[-1], 0, label="Target RT60")

    apply_decorations(plt, legend=True, location="lower left")
    
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [dB]")
    plt.ylim([-100, 6])
    plt.title(f"{file_name} RT60 Measurement: {est_rt60 * 1000:.0f} ms")                 
    save_plot(plt, file_name + "_RT60")


def plot_compare_waveform(target, output, sample_rate, title, xlim=None, ylim=None):
    target = target.numpy()
    output = output.numpy()

    target_ch, target_frames = target.shape
    output_ch, output_frames = output.shape
    assert target_ch == output_ch, "Both waveforms must have the same number of channels"
    
    time_target = torch.arange(0, target_frames) / sample_rate
    time_output = torch.arange(0, output_frames) / sample_rate

    figure, axes = plt.subplots(target_ch, 1, figsize=(10, 5))
    if target_ch == 1:
        axes = [axes]
    for c in range(target_ch):
        axes[c].plot(time_target, target[c], linewidth=1, alpha=0.8, label='Target')
        axes[c].plot(time_output, output[c], linewidth=1, alpha=0.8, label='Output')
        axes[c].grid(True)
        if target_ch > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        else:
            axes[c].set_ylabel('Amplitude')
        axes[c].set_xlabel('Time (s)')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
        apply_decorations(axes[c], legend=True)
    figure.suptitle(title)
    plt.tight_layout()
    plt.show(block=False)

    return figure