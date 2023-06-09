""" Impulse response and transfer function generator.
"""

import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft

from utils.signals import generate_reference
from utils.plotter import plot_impulse_response, plot_transfer_function
from inference import make_inference


def generate_impulse_response(load, sample_rate, device, duration):
    
    # Generate the reference signals
    sweep, inverse_filter, reference = generate_reference(args.duration, args.sample_rate) 
    # Make inference with the model on the sweep tone
    sweep_output = make_inference(
        args.load, sweep, args.sample_rate, args.device, max_length=None, stereo=False, tail=None, 
        width=50., c0=0., c1=0., gain_dB=0., mix=100.)

    # Assuming sweep_output is a torch.Tensor
    print("Before operation:")
    print("sweep_output shape:", sweep_output.shape)
    print("sweep_output dtype:", sweep_output.dtype)
    print("sweep_output device:", sweep_output.device)
    print("sweep_output data:\n", sweep_output)

    # Perform the operation
    sweep_output = sweep_output[0].cpu().numpy()

    print("\nAfter operation:")
    print("sweep_output shape:", sweep_output.shape)
    print("sweep_output dtype:", sweep_output.dtype)
    print("sweep_output data:\n", sweep_output)

    # Convolve the sweep tone with the inverse filter
    measured = np.convolve(sweep_output, inverse_filter)

    # Save and plot the measured impulse response
    save_as = f"{Path(load).stem}_IR.wav"
    wavfile.write(f"audio/processed/{save_as}", sample_rate, measured.astype(np.float32))
    print(f"Saved measured impulse response to {save_as}")

    # print(len(sweep))
    # print(len(inverse_filter))
    # print(len(measured))
    # print(type(sweep))
    # print(type(inverse_filter))
    # print(type(measured))

    plot_impulse_response(sweep_output, inverse_filter, measured, sample_rate, file_name=Path(load).stem)

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


def transfer_function(x: np.ndarray, y: np.ndarray):
    X = fft_scipy(x, len(y))
    Y = fft_scipy(y, len(y))
    tf = Y / X
    return tf

def generate_transfer_function(reference, measured_ir, sample_rate):
    tf = transfer_function(reference, measured_ir)
    magnitude = 20 * np.log10(np.abs(tf))
    phase = np.angle(tf) * 180 / np.pi
    return magnitude, phase


if __name__ == "__main__":
    parser = ArgumentParser(description='Generate impulse response or transfer function from a trained model')
    parser.add_argument('--load', type=str, required=True, help='Path (rel) to checkpoint to load')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate of the audio')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help='set device to run the model on')
    parser.add_argument("--duration", type=float, default=3.0, help="duration in seconds")
    parser.add_argument("--mode", type=str, choices=['ir', 'tf'], default='ir', help="Mode to run: 'ir' for impulse response or 'tf' for transfer function")

    args = parser.parse_args()

    if args.mode == 'ir':
        generate_impulse_response(args.load, args.sample_rate, args.device, args.duration)

    else: # mode == 'tf'
        measured_ir, reference = generate_impulse_response(args.load, args.sample_rate, args.device, args.duration)
        magnitude, phase = generate_transfer_function(reference, measured_ir, args.sample_rate)
        plot_transfer_function(magnitude, phase, args.sample_rate, file_name=Path(args.load).stem)

