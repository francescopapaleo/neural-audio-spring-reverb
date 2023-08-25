""" Impulse response and transfer function generator.
"""

import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.signals import generate_reference
from src.plotter import plot_impulse_response
from src.helpers import load_model_checkpoint
from inference import make_inference
from configurations import parse_args


def generate_impulse_response(checkpoint, sample_rate, device, duration, args):
    
    model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params = load_model_checkpoint(device, checkpoint, args)

    # Generate the reference signals
    sweep, inverse_filter, reference = generate_reference(duration, sample_rate) 
    sweep = sweep.reshape(1, -1)
    print("generated sweep:", sweep.shape)

    # Make inference with the model on the sweep tone
    sweep_output = make_inference(
        sweep, sample_rate, model, device, 100)

    # Perform the operation
    sweep_output = sweep_output[0].cpu().numpy()
    sweep_output = sweep_output.squeeze()
    print(f"sweep_output:{sweep_output.shape} || inverse_filter:{inverse_filter.shape}")

    # Convolve the sweep tone with the inverse filter
    measured = np.convolve(sweep_output, inverse_filter)
    
    measured_index = np.argmax(np.abs(measured))
    print(f"measured_index:{measured_index}")

    # Save and plot the measured impulse response
    save_as = f"{Path(checkpoint).stem}_IR.wav"
    wavfile.write(f"audio/proc/{save_as}", sample_rate, measured.astype(np.float32))
    print(f"Saved measured impulse response to {save_as}")

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
    for ax in axs:
        ax.grid(True)
        ax.set_xlabel('Time [sec]')

    axs[0].set_title(f'Model: {model_name} Impulse Response (IR)')
    axs[0].plot(measured)
    axs[0].set_ylim([-1, 1])
    
    axs[1].specgram(measured, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='hot', scale='dB')
    
    plt.tight_layout()
    plt.savefig(f"results/plots/{Path(checkpoint).stem}_IR.png")
    print(f"Saved measured impulse response plot to {Path(checkpoint).stem}_IR.png")
    plt.show()
        
    return sweep_output, measured

def fft_scipy(x: np.ndarray, fft_size: int, axis: int = -1) -> np.ndarray:

        if len(x) < fft_size:
            x_padded = np.pad(x, (0, fft_size - len(x)), 'constant', constant_values=(0,))
        else:
            x_padded = x[:fft_size]

        hM1 = int(np.floor((fft_size + 1) / 2))  # half analysis window size by rounding
        hM2 = int(np.floor(fft_size / 2))  # half analysis window size by floor
        hN = int((fft_size / 2) - 1)
        
        fft_buffer = np.zeros(shape=(fft_size,))  # initialize buffer for FFT
        fft_buffer[:hM1] = x_padded[hM2:]
        fft_buffer[-hM2:] = x_padded[:hM2]
        
        return fft(fft_buffer, fft_size, axis=axis)[:hN]

def transfer_function(x: np.ndarray, y: np.ndarray, n_fft: int, hop_length: int):
    len_x = len(x)
    len_y = len(y)
    
    # Ensure both signals have the same length
    if len_x > len_y:
        x = x[:len_y]
    elif len_y > len_x:
        y = y[:len_x]
    
    _, _, X = signal.stft(x, nperseg=n_fft, noverlap=hop_length)
    _, _, Y = signal.stft(y, nperseg=n_fft, noverlap=hop_length)
    tf = Y / X
    return tf

def generate_transfer_function(reference, measured_ir, n_fft: int, hop_length: int):
    tf = transfer_function(reference, measured_ir, n_fft, hop_length)
    magnitude = 20 * np.log10(np.abs(tf) / np.abs(tf).max())
    phase = np.angle(tf) * 180 / np.pi
    return magnitude, phase

from mpl_toolkits.mplot3d import Axes3D

def plot_tf_waterfall(magnitude, phase, sample_rate, n_fft, hop_length, file_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate time and frequency axes
    time_axis = np.arange(0, magnitude.shape[1] * hop_length / sample_rate, hop_length / sample_rate)
    freq_axis = np.fft.fftfreq(n_fft, d=1/sample_rate)[:n_fft//2]

    # Ensure dimensions match
    T, F = np.meshgrid(time_axis, freq_axis)
    magnitude = magnitude[:T.shape[0], :T.shape[1]]
    # magnitude = magnitude[:, :T.shape[1]]  # Trim magnitude to match shape

    # Plot the waterfall spectrogram
    surf = ax.plot_surface(T, F, magnitude, cmap='viridis', antialiased=True)

    ax.set_title('Transfer Function Waterfall Spectrogram')
    ax.set_xlabel('Time [sec]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_zlabel('Magnitude [dB]')

    # Rotate the view for better visibility
    ax.view_init(elev=30, azim=-45)

    plt.tight_layout()
    plt.savefig(f"results/plots/{file_name}.png")
    plt.show()


if __name__ == "__main__":
    print('Generate impulse response or transfer function from a trained model')

    args = parse_args()

    n_fft = 1024  # or any desired FFT size
    hop_length = n_fft // 2  # or any desired hop length

    if args.mode == 'ir':
        generate_impulse_response(args.checkpoint, args.sample_rate, args.device, args.duration, args)

    else: # mode == 'tf'
        measured_ir, reference = generate_impulse_response(args.checkpoint, args.sample_rate, args.device, args.duration, args)
        magnitude, phase = generate_transfer_function(reference, measured_ir, n_fft, hop_length)
        
        # Convert the string to a Path object
        checkpoint_path = Path(args.checkpoint)
        file_name = f'{checkpoint_path.stem}_waterfall'
        
        plot_tf_waterfall(magnitude, phase, args.sample_rate, n_fft, hop_length,file_name)
