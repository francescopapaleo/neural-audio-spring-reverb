""" Impulse response and transfer function generator.
"""

import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft
from scipy import signal

from src.signals import generate_reference
from src.plotter import plot_impulse_response
from src.helpers import load_audio, load_model_checkpoint
from inference import make_inference
from configurations import parse_args


def generate_impulse_response(checkpoint_path, sample_rate, device, duration):
    
    model, model_name, hparams = load_model_checkpoint(device, args.checkpoint_path)

    # Generate the reference signals
    sweep, inverse_filter, reference = generate_reference(duration, sample_rate) 

    x_p, fs_x, input_name = load_audio(sweep, args.sample_rate)

    # Make inference with the model on the sweep tone
    sweep_output = make_inference(
        x_p, model, device, args.c0, args.c1)

    # Perform the operation
    sweep_output = sweep_output[0].cpu().numpy()
    sweep_output = sweep_output.squeeze()
    print("sweep_output shape:", sweep_output.shape)
    print("inverse_filter shape:", inverse_filter.shape)

    # Convolve the sweep tone with the inverse filter
    measured = np.convolve(sweep_output, inverse_filter)

    # Save and plot the measured impulse response
    save_as = f"{Path(checkpoint_path).stem}_IR.wav"
    wavfile.write(f"audio/processed/{save_as}", sample_rate, measured.astype(np.float32))
    print(f"Saved measured impulse response to {save_as}")

    plot_impulse_response(sweep_output, inverse_filter, measured, sample_rate, file_name=Path(checkpoint_path).stem)

    return measured, reference

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


# def transfer_function(x: np.ndarray, y: np.ndarray):
#     X = fft_scipy(x, len(y))
#     Y = fft_scipy(y, len(y))
#     tf = Y / X
#     return tf

# def generate_transfer_function(reference, measured_ir, sample_rate):
#     tf = transfer_function(reference, measured_ir)
#     magnitude = 20 * np.log10(np.abs(tf))
#     phase = np.angle(tf) * 180 / np.pi
#     return magnitude, phase

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
    magnitude = 20 * np.log10(np.abs(tf) + 1e-10)
    phase = np.angle(tf) * 180 / np.pi
    return magnitude, phase


if __name__ == "__main__":
    print('Generate impulse response or transfer function from a trained model')

    args = parse_args()

    n_fft = 1024  # or any desired FFT size
    hop_length = n_fft // 2  # or any desired hop length

    if args.mode == 'ir':
        generate_impulse_response(args.checkpoint_path, args.sample_rate, args.device, args.duration)

    else: # mode == 'tf'
        measured_ir, reference = generate_impulse_response(args.checkpoint_path, args.sample_rate, args.device, args.duration)
        magnitude, phase = generate_transfer_function(reference, measured_ir, n_fft, hop_length)
        # plot_transfer_function(magnitude, phase, args.sample_rate, file_name=Path(args.checkpoint_path).stem)
    
    import matplotlib.pyplot as plt
    import numpy as np

    def plot_tf(tf, sample_rate, n_fft, hop_length, file_name):
        fig, ax = plt.subplots(figsize=(15, 7))

        # Compute the frequencies for each FFT bin
        freqs = np.linspace(0, sample_rate / 2, tf.shape[0])

        # Compute the times for each frame
        times = np.arange(tf.shape[1]) * hop_length / sample_rate

        # Compute 10 * log10 of the absolute value of the transfer function to convert to dB
        tf_dB = 10 * np.log10(np.abs(tf))

        im = ax.imshow(tf_dB, aspect='auto', origin='lower', extent=[times.min(), times.max(), freqs.min(), freqs.max()])
        ax.set_title('Spectrogram')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        fig.colorbar(im, ax=ax, format='%+2.0f dB')

        plt.tight_layout()
        plt.savefig(f"{file_name}_spectrogram.png")
        plt.show()

    plot_tf(magnitude, args.sample_rate, n_fft, hop_length, file_name=Path(args.checkpoint_path).stem)
