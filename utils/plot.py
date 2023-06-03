from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as T
import librosa

def plot_compare_waveform(y, y_pred):
    '''Plot the waveform of the input and the predicted signal.
    Parameters
    ----------
    y : array_like
        Ground truth signal
    y_pred : array_like
        The predicted signal
        '''

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    ax.plot(y, alpha=0.7, label='Ground Truth', color='blue')
    ax.plot(y_pred, alpha=0.7, label='Prediction', color='red')
    ax.set_title('Waveform Comparison (Ground Truth vs Prediction)')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()
    plt.savefig(Path(args.results_dir) / 'waveform_plot.png')
    plt.close(fig)
    print("Saved waveform plot to: ", Path(args.results_dir) / 'waveform_plot.png')


def plot_zoom_waveform(y, y_pred, sr, t_start=None, t_end=None, results_dir='./results'):
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
    t = np.arange(y.shape[0]) / sr

    # Determine the indices corresponding to the start and end times
    if t_start is not None:
        i_start = int(t_start * sr)
    else:
        i_start = 0

    if t_end is not None:
        i_end = int(t_end * sr)
    else:
        i_end = len(t)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))

    ax.plot(t[i_start:i_end], y[i_start:i_end], alpha=0.7, label='Ground Truth', color='blue')
    ax.plot(t[i_start:i_end], y_pred[i_start:i_end], alpha=0.7, label='Prediction', color='red')

    ax.set_title('Zoom on Waveform (Ground Truth vs Prediction)')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()

    plt.savefig(Path(results_dir) / 'waveform_zoom.png')
    print("Saved zoomed waveform plot to: ", Path(results_dir) / 'waveform_zoom.png')
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


def plot_compare_spectrogram(target, output, input, titles=['target', 'output', 'input'], ylabel="freq_bin", aspect="auto", xmax=None):
    '''Plot the spectrogram of the input, the prediction and the ground truth
    Parameters
    ----------
    target : array_like
        Ground truth spectrogram
    output : array_like
        The predicted spectrogram
    input : array_like
        The input spectrogram
    titles : list, optional
        List of titles for the subplots (default to ['target', 'output', 'input']).
    ylabel : str, optional
        Label for the y-axis (default to 'freq_bin').
    aspect : str, optional
        Aspect ratio of the plot (default to 'auto').
    xmax : int, optional
        Maximum value for the x-axis (default to None).
    '''
    spec1 = get_spectrogram(torch.Tensor(target))
    spec2 = get_spectrogram(torch.Tensor(output))
    spec3 = get_spectrogram(torch.Tensor(input))

    fig, axs = plt.subplots(1, 3, figsize=(15, 7)) # 1 row, 3 columns

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


def plot_transfer_function(magnitude, phase, sr, file_name):
    '''Plot the transfer function
    Parameters
    ----------
    magnitude : array_like
    The magnitude of the transfer function
    phase : array_like
    The phase of the transfer function
    sr : int
    The sampling frequency
    file_name : str
    The name of the file to save the plot to
    '''

    freqs = np.linspace(0, sr / 2, len(magnitude))
    
    fig, ax = plt.subplots(2, 1, figsize=(15, 7))
    ax[0].semilogx(freqs, magnitude)
    ax[0].set_xlim([1, freqs[-1]])
    ax[0].set_title("Transfer Function")
    ax[0].set_ylim([-100, 6])
    ax[0].set_xlabel("Frequency [Hz]")
    ax[0].set_ylabel("Magnitude [dBFS]")
    ax[0].grid(True)
    ax[1].semilogx(freqs, phase)
    ax[1].set_xlim([1, freqs[-1]])
    ax[1].set_ylim([-180, 180])
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Phase [degrees]")
    ax[1].grid(True)
    plt.suptitle("H(w) - Transfer Function")
    plt.tight_layout()
    plt.savefig(Path(args.results_dir) / file_name)
    plt.close(fig)
    print("Saved transfer function plot to: ", Path(args.results_dir) / file_name)


def plot_loss_function(loss_history, args):
    '''Plot the loss function over the training epochs
    Parameters
    ----------
    loss_history : list
        List of loss values over the training iterations
    args : Namespace
        Parsed command line arguments
    '''

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 7))

    ax.plot(loss_history, label='Loss')
    ax.set_title('Loss function over the training iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.grid(True)
    ax.legend()

    loss_function_path = Path(args.results_dir) / 'loss_plot.png'
    plt.savefig(loss_function_path)
    plt.close(fig)
    print("Saved loss function plot to: ", loss_function_path)


def plot_spectrogram(signals, sr, time_window):
        
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    for i, signal in enumerate(signals):
        axs[i].set_title(f'Spectrogram {i+1}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Frequency (Hz)')
        librosa.display.specshow(
            librosa.amplitude_to_db(
                np.abs(librosa.stft(signal, n_fft=512)),
                ref=np.max), sr=sr, 
                x_axis='time', y_axis='linear', 
                hop_length=512, ax=axs[i], 
                cmap='inferno')
        axs[i].axvspan(
            time_window[0], time_window[1], 
            alpha=0.5, color='r')
        plt.show()


def plot_specgram(signals, sr, time_window, 
                  title="Spectrogram", filename="spectrogram"):
    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    for i, signal in enumerate(signals):
        axs[i].specgram(signal, NFFT=64, Fs=sr, 
                        noverlap=32, cmap='inferno', vmin=-60, vmax=0)
        axs[i].set_title(f'Spectrogram {i+1}')
        axs[i].set_xlabel('Time (s)')
        axs[i].set_ylabel('Frequency (Hz)')
        axs[i].set_ylim([0, 5000])
        axs[i].set_xlim(time_window)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('./imgs/'+ filename +'.png')
    plt.show()

def plot_sweep_inverse_measured(sweep_filt, inverse_filter, 
                                 measured, duration, file_name):
    '''Plot the sweep tone, the inverse filter and the measured impulse response
        Parameters
        ----------
        sweep_filt : array_like
        The sweep tone
        inverse_filter : array_like
        The inverse filter
        measured : array_like
        The measured impulse response
        sr : int
        The sampling frequency
        duration : float
        The duration of the sweep tone  
        file_name : str
        The name of the file to save the plot to
        '''
    time_stamps = np.arange(0, duration, 1/ args.sr)
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
    
    time_stamps = np.arange(0, len(measured)/ args.sr, 1/ args.sr)
    ax[2].plot(time_stamps, measured)
    ax[2].set_xlim([0, time_stamps[-1]])
    ax[2].set_title("Impulse Response")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(Path(args.results_dir) / file_name)
    plt.close(fig)
    print("Saved signal plot to: ", Path(args.results_dir) / file_name)

def plot_metrics(test_results, args):
    '''Plot the metrics over the test set'''
    time_values = np.arange(len(test_results['mse']))
    
    fig, ax = plt.subplots(figsize=(15, 7))
    for metric_name, values in test_results.items():
        ax.plot(time_values, values, label=metric_name.capitalize())

    ax.set_xlabel("Sample")
    ax.set_ylabel("Normalized Metric Value")
    ax.set_title("Evaluation: Metrics Over Test Set")
    ax.legend()
    ax.grid(True)
    plt.savefig(Path(args.results_dir) / 'eval_metrics_plot.png')


def plot_input_target(**args):

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
    axs[2].legend()
    axs[2].grid(True)


    # Plot target spectrogram
    axs[1].imshow(10 * np.log10(y), aspect='auto', origin='lower')
    axs[1].set_title('Target Spectrogram')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    axs[1].colorbar(label='dB')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(args.results_dir) / 'dataset_spectrum.png')

def save_plot(plt, path, filename):
    plt.savefig(Path(path) / filename)

if __name__ == '__main__':
    args = parser.parse_args()