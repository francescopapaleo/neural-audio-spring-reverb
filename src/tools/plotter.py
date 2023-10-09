import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import spectrogram
from pathlib import Path


def save_plot(plt, file_name, args):
    plot_dir = Path(args.plots_dir)
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


def generate_spectrogram(waveform, sample_rate):
    frequencies, times, Sxx = spectrogram(
        waveform, 
        fs=sample_rate, 
        window='hann',
        nperseg=32,
        noverlap=16,  
        scaling='spectrum', 
        mode='magnitude'
    )
    
    # Convert magnitude to dB
    Sxx_dB = 10 * np.log10(Sxx + 1e-10)
    
    return frequencies, times, Sxx_dB



def plot_waterfall(waveform, title, sample_rate, args, stride=1):
    frequencies, times, Sxx = generate_spectrogram(waveform, sample_rate)

    fig = plt.figure(figsize=(10, 10))
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
    ax.autoscale()  # Adjusts the viewing limits for better visualization
    cbar = fig.colorbar(surf, ax=ax, pad=0.01, aspect=35, shrink=0.5)
    cbar.set_label('Magnitude (dB)')

    # Set labels and title
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Time (seconds)')
    ax.set_zlabel('Magnitude (dB)')
    
    ax.set_xlim([frequencies[-1], frequencies[0]])
    # ax.set_xscale('symlog', linthreshx=0.01)
    # ax.set_xscale('log')
    # ax.set_xlim([20000, 20])  # Set the x-axis limit to be between 20 and 20,000 Hz in log scale
    ax.view_init(elev=10, azim=45, roll=None, vertical_axis='z')  # Adjusts the viewing angle for better visualization
    save_plot(fig, title, args)


def plot_ir_spectrogram(signal, sample_rate, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))

    # Calculate the duration of the signal in seconds
    duration_seconds = len(signal) / sample_rate

    ax.set_title(title)
    
    # Plot the spectrogram
    cax = ax.specgram(signal, NFFT=512, Fs=sample_rate, noverlap=256, cmap='hot', scale='dB', mode='magnitude', vmin=-100, vmax=0)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel(f'Time [sec] ({duration_seconds:.2f} s)')  # Label in seconds
    ax.grid(True)

    # Add the colorbar
    cbar = fig.colorbar(mappable=cax[3], ax=ax, format="%+2.0f dB")
    cbar.set_label('Intensity [dB]')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved spectrogram plot to {save_path}")



def plot_rt60(T, energy_db, e_5db, est_rt60, rt60_tgt, file_name, args):
    plt.subplots(figsize=(5, 5))
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
    save_plot(plt, file_name + "_RT60", args)


def plot_signals(sweep: np.ndarray, inverse_filter: np.ndarray, measured: np.ndarray, sample_rate: int, file_name: str, args):
    fig, ax = plt.subplots(3, 1, figsize=(10,5))
    plot_data(get_time_stamps_np(len(sweep), sample_rate), sweep, ax[0], "Processed Sweep Tone", "Time [s]", "Amplitude")
    plot_data(get_time_stamps_np(len(inverse_filter), sample_rate), inverse_filter, ax[1], "Inverse Filter", "Time [s]", "Amplitude")
    plot_data(get_time_stamps_np(len(measured), sample_rate), measured, ax[2], "Impulse Response", "Time [s]", "Amplitude")
    fig.suptitle(f"{file_name} - Impulse Response δ(t)")
    save_plot(fig, file_name + "_IR", args)
