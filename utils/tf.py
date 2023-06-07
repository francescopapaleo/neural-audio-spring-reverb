# Description: Computes the transfer function of the impulse response

from argparse import ArgumentParser
from pathlib import Path
from scipy.fft import fft
import numpy as np
import torch

from inference import make_inference
from utils.generator import generate_reference
from utils.plotter import plot_transfer_function


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

def compute_tf(x: np.ndarray, y: np.ndarray):
    X = fft_scipy(x, len(x)) # if isinstance(x, np.ndarray) else fft_scipy(x.numpy(), len(x))
    Y = fft_scipy(y, len(x)) # if isinstance(y, np.ndarray) else fft_scipy(y.numpy(), len(y))
    return Y / X  # Y(w) / X(w)

def tf_main(load, 
            input_path,
            sample_rate,
            device):

    y_hat = make_inference(load, input_path, sample_rate, device, max_length=None, stereo=False, tail=None, width=50., c0=0., c1=0., gain_dB=0., mix=100.)

    print("Computing transfer function...")

    # measured = y_hat[0].cpu().numpy()

    # Compute the transfer function
    # tf_measured = compute_tf(reference, measured)
    # magnitude = 20 * np.log10(np.abs(tf_measured))
    # phase = np.angle(tf_measured) * 180 / np.pi
    
    # plot_transfer_function(magnitude, phase, sample_rate, "transfer_function.png")

    # return tf_measured

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=16000, help="sample rate")
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    parser.add_argument("--duration", type=float, default=5.0, help="duration in seconds")

    parser.add_argument('--max_length', type=float, default=None, help='maximum length of the output audio')
    parser.add_argument('--stereo', type=bool, default=False, help='flag to indicate if the audio is stereo or mono')
    parser.add_argument('--tail', type=bool, default=False, help='flag to indicate if tail padding is required')
    parser.add_argument('--width', type=float, default=50, help='width parameter for the model')
    parser.add_argument('--c0', type=float, default=0, help='c0 parameter for the model')
    parser.add_argument('--c1', type=float, default=0, help='c1 parameter for the model')
    parser.add_argument('--gain_dB', type=float, default=0, help='gain in dB for the model')
    parser.add_argument('--mix', type=float, default=50, help='mix parameter for the model')

    args = parser.parse_args()

    args.input_path = Path('./data/raw/reference.wav')
    args.load = 'tcn_25_16_0.001_20230605_184451.pt'
    tf_main(args.load, args.input_path, args.sample_rate, args.device)
      