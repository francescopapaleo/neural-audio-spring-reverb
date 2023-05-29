# Description: Computes the transfer function of the impulse response
from config import parser
from pathlib import Path

import scipy.signal
import scipy.fftpack
import numpy as np

from inference import make_inference
from utils.plot import plot_transfer_function
from utils.rt60 import measure_rt60
from utils.generator import generate_reference

args = parser.parse_args()
sample_rate = args.sr


def fft_scipy(x: np.ndarray, fft_size: int, axis: int = -1) -> np.ndarray:
        # Pad x with zeros at the end if padding_size is positive
        # padding_size = fft_size - len(x)
        # if padding_size > 0:
        #     x_padded = np.pad(x, (0, padding_size), 'constant', constant_values=(0,))
        # else:
        #     x_padded = x

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
        
        return scipy.fftpack.fft(fft_buffer, fft_size, axis=axis)[:hN]

def compute_tf(x: np.ndarray, y: np.ndarray):
    X = fft_scipy(x, len(x)) # if isinstance(x, np.ndarray) else fft_scipy(x.numpy(), len(x))
    Y = fft_scipy(y, len(x)) # if isinstance(y, np.ndarray) else fft_scipy(y.numpy(), len(y))
    return Y / X  # Y(w) / X(w)

def tf_main(duration: float = 5.0):

    # Generate the reference signal
    sweep, inverse_filter, reference = generate_reference(duration)
    
    print(type(inverse_filter))  # Check its type

    y_hat = make_inference()

    print("Computing transfer function...")

    measured = y_hat[0].numpy()

    # Compute the transfer function
    tf_measured = compute_tf(reference, measured)
    magnitude = 20 * np.log10(np.abs(tf_measured))
    phase = np.angle(tf_measured) * 180 / np.pi
    
    plot_transfer_function(magnitude, phase, sample_rate, "transfer_function.png")

    return tf_measured

if __name__ == "__main__":
    tf_measured = tf_main(5.0)
