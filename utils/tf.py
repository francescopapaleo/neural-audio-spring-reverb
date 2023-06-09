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
from utils.generator import generate_reference
from utils.plotter import plot_transfer_function, plot_ir


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
    plot_ir(sweep_test, inverse_filter, measured_ir, sample_rate, file_name)
    
    measured_output_path = Path('./audio/processed')  / (file_name + '.wav')
    wavfile.write(measured_output_path, sample_rate, reference.astype(np.float32))

    # Compute the transfer function
    tf = transfer_function(reference, measured_ir)
    magnitude = 20 * np.log10(np.abs(tf))
    phase = np.angle(tf) * 180 / np.pi
    
    plot_transfer_function(magnitude, phase, sample_rate, file_name)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=16000, help="sample rate")
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--load', type=str, default=None, help='relative path to checkpoint to load')
    parser.add_argument("--duration", type=float, default=2.0, help="duration in seconds")

    parser.add_argument('--max_length', type=float, default=None, help='maximum length of the output audio')
    parser.add_argument('--stereo', type=bool, default=False, help='flag to indicate if the audio is stereo or mono')
    parser.add_argument('--tail', type=bool, default=False, help='flag to indicate if tail padding is required')
    parser.add_argument('--width', type=float, default=50, help='width parameter for the model')
    parser.add_argument('--c0', type=float, default=0, help='c0 parameter for the model')
    parser.add_argument('--c1', type=float, default=0, help='c1 parameter for the model')
    parser.add_argument('--gain_dB', type=float, default=0, help='gain in dB for the model')
    parser.add_argument('--mix', type=float, default=50, help='mix parameter for the model')

    args = parser.parse_args()

    for file in Path('./checkpoints').glob('tcn_*'):
        args.load = file

        load = args.load
        sample_rate = args.sample_rate
        duration = args.duration
        device = args.device

        main(load, sample_rate, device, duration)


    """ Generate the reference signal and plot the transfer function
    sweep, inverse_filter, measured_ref = generate_reference(duration, sample_rate) 
    
    tf = transfer_function(sweep, inverse_filter, measured_ref)
    magnitude = 20 * np.log10(np.abs(tf))
    phase = np.angle(tf) * 180 / np.pi

    plot_transfer_function(magnitude, phase, sample_rate, 'generator_reference')
    """