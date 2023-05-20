import numpy as np
from scipy.io.wavfile import write
from config import *
from utils.plot import plot_signals 
from pathlib import Path

def sine(SAMPLE_RATE: int, 
         duration: float, 
         amplitude: float, 
         frequency: float = 440.0) -> np.ndarray:
    N = np.arange(0, duration, 1.0 / SAMPLE_RATE)
    return amplitude * np.sin(2.0 * np.pi * frequency * N)


def sweep_tone(SAMPLE_RATE: int, 
               duration: float, 
               amplitude: float, 
               f0: float = 20, 
               f1: float = 20000, 
               inverse: bool = False) -> np.ndarray:
    R = np.log(f1 / f0)
    t = np.arange(0, duration, 1.0 / SAMPLE_RATE)
    output = np.sin((2.0 * np.pi * f0 * duration / R) * (np.exp(t * R / duration) - 1))
    if inverse:
        k = np.exp(t * R / duration)
        output = output[::-1] / k
    return amplitude * output


def generate_reference(duration: float = 5.0):
    decibels = -18
    amplitude = 10 ** (decibels / 20)
    f0 = 5
    f1 = SAMPLE_RATE / 2

    sweep = sweep_tone(SAMPLE_RATE, duration, amplitude, f0=f0, f1=f1)
    
    inverse_filter = sweep_tone(SAMPLE_RATE, duration, amplitude, f0=f0, f1=f1, inverse=True)
    
    N = len(sweep)
    reference = np.convolve(inverse_filter, sweep, mode='same')

    # Scale to int16
    sweep_int16 = np.int16(sweep / np.max(np.abs(sweep)) * 32767)
    inverse_int16 = np.int16(inverse_filter / np.max(np.abs(inverse_filter)) * 32767)
    reference_int16 = np.int16(reference / np.max(np.abs(reference)) * 32767)

    # Save as .wav files
    sweep_output_path = AUDIO_DIR / "sweep.wav"
    write(sweep_output_path, SAMPLE_RATE, sweep_int16)
    
    inverse_output_path = AUDIO_DIR / "inverse_filter.wav"
    write(inverse_output_path, SAMPLE_RATE, inverse_int16)

    reference_output_path = AUDIO_DIR / "reference.wav"
    write(reference_output_path, SAMPLE_RATE, reference_int16)

    return sweep_int16, inverse_int16, reference_int16


if __name__ == "__main__":
    duration = 5
    
    # Generate and save the impulse, sine wave, and sweep tone
    sweep, inverse_filter, reference = generate_reference(duration)

    # Plot them
    plot_signals(sweep, inverse_filter, reference, SAMPLE_RATE, duration, "reference.png")