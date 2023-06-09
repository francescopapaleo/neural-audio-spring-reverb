""" Signal Generators for Measurements

Modified version from the originally written by Xavier Lizarraga
"""

import numpy as np
from scipy.io import wavfile
from typing import Tuple
from pathlib import Path
from argparse import ArgumentParser
from utils.plotter import plot_impulse_response


def impulse(sample_rate: int, duration: float, file_name: str = "impulse") -> np.ndarray:
    '''Generate an impulse

    Arguments:
    ----------
    sample_rate (int): Sample rate for the audio file.
    array_length (int): Total length of the array.
    file_name (str): Name of the audio file to be saved.
    '''
    array_length = int(duration * sample_rate)
    impulse = np.zeros(array_length)
    impulse[0] = 1.0  # Place the impulse at the first sample

    return impulse


def sine(sample_rate: int, duration: float, amplitude: float, frequency: float = 440.0) -> np.ndarray:
    '''Generate a sine wave
    
    Arguments:
    ----------
        sample_rate (int): Sample rate.
        duration (float): Duration of the sine wave.
        amplitude (float): Amplitude of the sine wave.
        frequency (float, optional): Frequency of the sine wave. Defaults to 440Hz.
    '''
    N = np.arange(0, duration, 1.0 / sample_rate)
    sine = amplitude * np.sin(2.0 * np.pi * frequency * N)
    return sine


def sweep_tone(sample_rate: int, duration: float, amplitude: float, f0: float = 20, f1: float = 20000, inverse: bool = False) -> np.ndarray:
    '''Generate a sweep tone
    
    Arguments:
    ----------
        sample_rate (int): Sample rate.
        duration (float): Duration of the tone.
        amplitude (float): Amplitude of the tone.
        f0 (float, optional): Start frequency, defaults to 20Hz.
        f1 (float, optional): End frequency, defaults to 20kHz.
        inverse (bool, optional): Generate inverse filter, defaults to False.
    '''
    R = np.log(f1 / f0)
    t = np.arange(0, duration, 1.0 / sample_rate)
    output = np.sin((2.0 * np.pi * f0 * duration / R) * (np.exp(t * R / duration) - 1))
    if inverse:
        k = np.exp(t * R / duration)
        output = output[::-1] / k
    sweep_tone = amplitude * output
    return sweep_tone


def generate_reference(duration: float, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Generate the reference impulse response

    Arguments:
    ----------
        duration (float): Duration of the tone.
        sample_rate (int): Sample rate.

    Returns:
    --------
        sweep (np.ndarray): The sweep tone.
        inverse_filter (np.ndarray): The inverse filter.
        reference (np.ndarray): The reference impulse response.
    '''
    decibels = -18
    amplitude = 10 ** (decibels / 20)
    f0 = 20
    f1 = sample_rate / 2

    # Generate the sweep tone and inverse filter
    sweep = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1)
    inverse_filter = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1, inverse=True)
    
    # Convolves the sweep tone with the inverse filter in order to obtain the impulse response x(t)
    N = len(sweep)
    reference = np.convolve(inverse_filter, sweep)

    return sweep, inverse_filter, reference


def save_audio(path: str, sample_rate: int, audio: np.ndarray, ):
    '''Saves an audio array to a .wav file.

    Arguments:
    ----------
        path (str): File path for the audio file to be saved.
        sample_rate (int): Sample rate.
        audio (np.ndarray): Audio array.
    '''
    output_path = Path('./data/generated') / f"{path}.wav"
    wavfile.write(output_path, sample_rate, audio.astype(np.float32))


def main(duration: float, sample_rate: int):

    # Generate the arrays
    sweep, inverse_filter, reference = generate_reference(duration, sample_rate)
    single_impulse = impulse(sample_rate, duration)

    # Save as .wav files
    save_audio("sweep", sample_rate, sweep)
    save_audio("inverse_filter", sample_rate, inverse_filter)
    save_audio("generator_reference", sample_rate, reference)
    save_audio("single_impulse", sample_rate, single_impulse)

    # Plot them
    plot_impulse_response(sweep, inverse_filter, reference, args.sample_rate, "generator_reference")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate audio files for measurements")
    parser.add_argument("--duration", type=float, default=3.0, help="duration in seconds")
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate of the audio')

    args = parser.parse_args()

    main(args.duration, args.sample_rate)
    