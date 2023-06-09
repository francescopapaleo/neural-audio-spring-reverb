""" Impulse response generator.
"""

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
    
    print(f"sweep test shape: {sweep_test.shape} - dtype: {sweep_test.dtype}")        
    
    sweep_test = sweep_test[0].cpu().numpy()
    measured_ir = np.convolve(sweep_test, inverse_filter)

    save_as = f"{Path(load).stem}_IR.wav"
    wavfile.write(f"audio/processed/{save_as}", sample_rate, measured_ir.astype(np.float32))

    plot_ir(sweep_test, inverse_filter, measured_ir, sample_rate, file_name=Path(load).stem)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--load', type=str, default=None, help='relative path to checkpoint to load')
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0, help="duration in seconds")

    args = parser.parse_args()

    for file in Path('./checkpoints').glob('tcn_*'):
        args.load = file

        load = args.load
        sample_rate = args.sample_rate
        duration = args.duration
        device = args.device

        impulse_response(args.load, args.sample_rate, args.device, args.duration)