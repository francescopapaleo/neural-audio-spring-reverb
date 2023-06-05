# Description: Generate the reference signal for the experiment
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from scipy.io.wavfile import write
from utils.plot import plot_signals 

parser = ArgumentParser()
args = parser.parse_args()
sample_rate = args.sr

def sine(sample_rate: int, 
         duration: float, 
         amplitude: float, 
         frequency: float = 440.0) -> np.ndarray:
    N = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * np.sin(2.0 * np.pi * frequency * N)


def sweep_tone(sample_rate: int, 
               duration: float, 
               amplitude: float, 
               f0: float = 20, 
               f1: float = 20000, 
               inverse: bool = False) -> np.ndarray:
    R = np.log(f1 / f0)
    t = np.arange(0, duration, 1.0 / sample_rate)
    output = np.sin((2.0 * np.pi * f0 * duration / R) * (np.exp(t * R / duration) - 1))
    if inverse:
        k = np.exp(t * R / duration)
        output = output[::-1] / k
    return amplitude * output


def generate_reference(duration: float = 5.0):
    decibels = -18
    amplitude = 10 ** (decibels / 20)
    f0 = 5
    f1 = sample_rate / 2

    sweep = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1)
    
    inverse_filter = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1, inverse=True)
    
    N = len(sweep)
    reference = np.convolve(inverse_filter, sweep, mode='same')

    # Scale to int16
    sweep_int16 = np.int16(sweep / np.max(np.abs(sweep)) * 32767)
    inverse_int16 = np.int16(inverse_filter / np.max(np.abs(inverse_filter)) * 32767)
    reference_int16 = np.int16(reference / np.max(np.abs(reference)) * 32767)

    # Save as .wav files
    sweep_output_path = Path(args.results_dir) / "sweep.wav"
    write(sweep_output_path, sample_rate, sweep_int16)
    
    inverse_output_path = Path(args.results_dir) / "inverse_filter.wav"
    write(inverse_output_path, sample_rate, inverse_int16)

    reference_output_path = Path(args.results_dir) / "reference.wav"
    write(reference_output_path, sample_rate, reference_int16)

    return sweep_int16, inverse_int16, reference_int16

def main():
    duration = 5
    # Generate and save the impulse, sine wave, and sweep tone
    sweep, inverse_filter, reference = generate_reference(duration)

    # Plot them
    plot_signals(sweep, inverse_filter, reference, sample_rate, duration, "reference.png")


if __name__ == "__main__":
    main()