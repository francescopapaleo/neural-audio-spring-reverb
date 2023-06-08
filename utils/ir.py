import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft
from utils.generator import generate_reference
from utils.plotter import plot_transfer_function, plot_ir
from inference import make_inference


def impulse_response(load, sample_rate, device, duration):

    sweep, inverse_filter, measured_ref = generate_reference(duration, sample_rate) 

    sweep_test = make_inference(load, sweep, sample_rate, device, max_length=None, stereo=False, tail=None, width=50., c0=0., c1=0., gain_dB=0., mix=100.)
        
    sweep_test = sweep_test[0].cpu().numpy()
    measured_ir = np.convolve(sweep_test, inverse_filter)

    wavfile.write(f"{load}ir.wav", sample_rate, measured_ir.astype(np.float32))

    plot_ir(sweep_test, inverse_filter, measured_ir, sample_rate, load)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=16000, help="sample rate")
    parser.add_argument("--device", type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--load", type=str, help="model name to load")
    parser.add_argument("--duration", type=float, default=5.0, help="duration in seconds")

    args = parser.parse_args()

    impulse_response(args.load, args.sample_rate, args.device, args.duration)