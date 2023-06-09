""" Transfer Function

Modified version from the originally written by Xavier Lizarraga
"""

from argparse import ArgumentParser
from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft
import numpy as np
import torch

from inference import make_inference
from utils.signals import generate_reference
from utils.plotter import plot_transfer_function, plot_impulse_response


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

def main(load, sample_rate, device, duration):

    sweep, inverse_filter, reference = generate_reference(duration, sample_rate) 
    
    sweep_test = make_inference(load, sweep, sample_rate, device, 
                                max_length=None, stereo=False, tail=None, width=50., c0=0., c1=0., gain_dB=0., mix=100.)
    
    sweep_test = sweep_test[0].cpu().numpy()
    measured_ir = np.convolve(sweep_test, inverse_filter)

    # Plot the impulse response
    file_name = f'{Path(load).stem}'
    plot_impulse_response(sweep_test, inverse_filter, measured_ir, sample_rate, file_name)
    
    measured_output_path = Path('./audio/processed')  / (file_name + '.wav')
    wavfile.write(measured_output_path, sample_rate, reference.astype(np.float32))

    # Compute the transfer function
    tf = transfer_function(reference, measured_ir)
    magnitude = 20 * np.log10(np.abs(tf))
    phase = np.angle(tf) * 180 / np.pi
    
    plot_transfer_function(magnitude, phase, sample_rate, file_name)

if __name__ == "__main__":
    parser = ArgumentParser(description='Transfer Function')
    parser.add_argument('--load', type=str, required=True, help='Path (rel) to checkpoint to load')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate of the audio')
    parser.add_argument('--device', type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", help='set device to run the model on')
    parser.add_argument("--duration", type=float, default=3.0, help="duration in seconds")

    args = parser.parse_args()


    main(args.load, args.sample_rate, args.device, args.duration)

