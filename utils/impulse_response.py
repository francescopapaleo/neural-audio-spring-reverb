""" Impulse response generator.
"""

import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft

from utils.signals import generate_reference
from utils.plotter import plot_impulse_response
from inference import make_inference


def generate_impulse_response(load, sample_rate, device, duration):

    # Generate the reference signals
    sweep, inverse_filter, reference = generate_reference(duration, sample_rate) 

    # Make inference with the model on the sweep tone
    sweep_test = make_inference(load, sweep, sample_rate, device, max_length=None, stereo=False, tail=None, width=50., c0=0., c1=0., gain_dB=0., mix=100.)
    
    # Convert to numpy array
    sweep_test = sweep_test[0].cpu().numpy()

    # Convolve the sweep tone with the inverse filter
    measured_ir = np.convolve(sweep_test, inverse_filter)

    # Save and plot the measured impulse response
    save_as = f"{Path(load).stem}_IR.wav"
    wavfile.write(f"audio/processed/{save_as}", sample_rate, measured_ir.astype(np.float32))
    plot_impulse_response(sweep_test, inverse_filter, measured_ir, sample_rate, file_name=Path(load).stem)


if __name__ == "__main__":
    parser = ArgumentParser(description='Generate impulse response from a trained model')
    parser.add_argument('--load', type=str, required=True, help='Path (rel) to checkpoint to load')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate of the audio')
    parser.add_argument('--device', type=str, 
                            default="cuda:0" if torch.cuda.is_available() else "cpu", help='set device to run the model on')
    parser.add_argument("--duration", type=float, default=3.0, help="duration in seconds")

    args = parser.parse_args()

    generate_impulse_response(args.load, args.sample_rate, args.device, args.duration)