from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torchaudio.transforms as T


def save_plot(plt, file_name):
    plot_dir = Path('./data/plots')
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


def get_time_stamps(signal_length, sample_rate):
    return np.arange(0, signal_length / sample_rate, 1 / sample_rate)


def plot_ir(sweep: np.ndarray, inverse_filter: np.ndarray, measured: np.ndarray, sample_rate: int, file_name: str):
    fig, ax = plt.subplots(3, 1, figsize=(15,7))
    plot_data(get_time_stamps(len(sweep), sample_rate), sweep, ax[0], "Sweep Tone", "Time [s]", "Amplitude")
    plot_data(get_time_stamps(len(inverse_filter), sample_rate), inverse_filter, ax[1], "Inverse Filter", "Time [s]", "Amplitude")
    plot_data(get_time_stamps(len(measured), sample_rate), measured, ax[2], "Impulse Response", "Time [s]", "Amplitude")
    fig.suptitle(f"{file_name} - Impulse Response δ(t)")
    save_plot(fig, file_name + "_IR")


def plot_transfer_function(magnitude: np.ndarray, phase: np.ndarray, sample_rate: int, file_name: str):
    freqs = np.linspace(0, sample_rate / 2, len(magnitude))
    fig, ax = plt.subplots(2, 1, figsize=(15, 7))
    plot_data(freqs, magnitude, ax[0], "Transfer Function", "Frequency [Hz]", "Magnitude [dBFS]")
    plot_data(freqs, phase, ax[1], " ", "Frequency [Hz]", "Phase [degrees]")
    plt.suptitle(f"{file_name} - Transfer Function H(w)")
    save_plot(fig, file_name + "_TF")


def plot_rt60(T, energy_db, e_5db, est_rt60, rt60_tgt, file_name):
    plt.subplots(figsize=(15, 4))
    plt.plot(T, energy_db, label="Energy")
    plt.plot([0, est_rt60], [e_5db, -65], "--", label="Linear Fit")
    plt.plot(T, np.ones_like(T) * -60, "--", label="-60 dB")
    plt.vlines(
        est_rt60, energy_db[-1], 0, linestyles="dashed", label="Estimated RT60"
    )

    if rt60_tgt is not None:
        plt.vlines(rt60_tgt, energy_db[-1], 0, label="Target RT60")

    apply_decorations(plt, legend=True, location="lower left")
    # plt.text(
    #     est_rt60, energy_db[-1] - 2, f"Estimated RT60: {est_rt60 * 1000:.0f} ms",
    #     ha="center", fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
    # )

    # plt.annotate(
    #         f"RT60: {est_rt60 * 1000:.0f} ms",
    #         xy=(est_rt60, energy_db[-1]),
    #         xytext=(est_rt60 - 0.25, energy_db[-1] + 50),
    #         arrowprops=dict(arrowstyle="->"),
    #         ha="right"
    #     )

    plt.xlabel("Time [s]")
    plt.ylabel("Energy [dB]")
    plt.ylim([-100, 6])
    plt.title(f"{file_name} RT60 Measurement: {est_rt60 * 1000:.0f} ms")                 
    save_plot(plt, file_name + "_RT60")

def plot_generic_waveform(y, y_pred, title, file_name):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    ax.plot(y, alpha=0.7, label='Ground Truth', color='blue')
    ax.plot(y_pred, alpha=0.7, label='Prediction', color='red')
    ax.set_title(title)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    apply_decorations(ax, legend=True)
    save_plot(fig, file_name)


def plot_compare_waveform(y, y_pred, file_name):
    plot_generic_waveform(y, y_pred, 'Waveform Comparison (Ground Truth vs Prediction)', file_name)


def plot_zoom_waveform(y, y_pred, sample_rate: int, file_name: str, t_start=None, t_end=None):
    t = np.arange(y.shape[0]) / sample_rate
    i_start = int(t_start * sample_rate) if t_start is not None else 0
    i_end = int(t_end * sample_rate) if t_end is not None else len(t)
    plot_generic_waveform(y[i_start:i_end], y_pred[i_start:i_end], 'Zoom on Waveform (Ground Truth vs Prediction)', file_name)


def get_spectrogram(waveform, n_fft=400, win_len=None, hop_len=None, power=2.0):
    spectrogram = T.Spectrogram(n_fft=n_fft, win_length=win_len, hop_length=hop_len, center=True, pad_mode="reflect", power=power,)
    return spectrogram(waveform)


def plot_compare_spectrogram(target, output, input, file_name: str, titles=['target', 'output', 'input'], ylabel="freq_bin", aspect="auto", xmax=None):
    specs = [get_spectrogram(torch.Tensor(sig)) for sig in [target, output, input]]
    fig, axs = plt.subplots(1, 3, figsize=(15, 7)) 

    for idx, spec in enumerate(specs):
        axs[idx].set_title(titles[idx])
        axs[idx].set_ylabel(ylabel)
        axs[idx].set_xlabel("frame")
        im = axs[idx].imshow(torch.log10(spec[0]), origin="lower", aspect=aspect, cmap='hot')
        if xmax:
            axs[idx].set_xlim((0, xmax))
        fig.colorbar(im, ax=axs[idx])

    save_plot(plt, file_name)


