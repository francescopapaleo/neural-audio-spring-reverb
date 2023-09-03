import numpy as np
import torchaudio
import torch
import os
from typing import Tuple
from pathlib import Path
import matplotlib.pyplot as plt

from src.default_args import parse_args
from src.tools.plotter import plot_data, save_plot, get_time_stamps_np

""" 
Signal Generators for Measurements
===================================
This module contains functions to generate the signals used in the measurements:
    - Impulse
    - Sine wave
    - Sweep tone
    - Reference impulse response

Modified version from the originally written by Xavier Lizarraga
"""

def impulse(sample_rate: int, duration: float, decibels: float = -1.0) -> np.ndarray:
    '''
    Generate an impulse

    Arguments:
    ----------
        sample_rate (int): Sample rate for the audio file.
        duration (float): Duration of the impulse.
        decibels (float): Amplitude of the impulse in decibels.
        file_name (str): Name of the audio file to be saved.
    '''
    array_length = int(duration * sample_rate)
    impulse = np.zeros(array_length)
    impulse[0] = 10 ** (decibels / 20)  # Convert decibels to amplitude

    return impulse


def sine(sample_rate: int, duration: float, amplitude: float, frequency: float = 440.0) -> np.ndarray:
    '''
    Generate a sine wave
    
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
    '''
    Generate a logaritmic sweep tone
    
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


def generate_reference(duration: float, sample_rate: int, decibels: float = -18, f0: float = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Generate the reference impulse response

    Arguments:
    ----------
        duration (float): Duration of the tone.
        sample_rate (int): Sample rate.
        decibels (float): The decibel level of the signal. Defaults to -18 dB.
        f0 (float): The start frequency of the sweep. Defaults to 20Hz.

    Returns:
    --------
        sweep (np.ndarray): The sweep tone.
        inverse_filter (np.ndarray): The inverse filter.
        reference (np.ndarray): The reference impulse response.
    '''
    amplitude = 10 ** (decibels / 20)
    f1 = sample_rate / 2

    # Generate the sweep tone and inverse filter
    sweep = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1)
    inverse_filter = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1, inverse=True)
    
    # Convolves the sweep tone with the inverse filter in order to obtain the impulse response x(t)
    N = len(sweep)
    impulse_response = np.convolve(inverse_filter, sweep)

    return sweep, inverse_filter, impulse_response


def save_audio(dir_path: str, file_name: str, sample_rate: int, bit_rate:int, waveform: np.ndarray):
    '''
    Save an audio array to a .wav file.

    Arguments:
    ----------
        dir_path (str): Directory path for the audio file to be saved.
        file_name (str): Name of the audio file to be saved.
        sample_rate (int): Sample rate.
        audio (np.ndarray): Audio array.
    '''
    # Create the directory if it does not exist
    output_directory = Path(dir_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{file_name}.wav"
    
    # Save the audio file
    waveform = waveform[np.newaxis, :]

    waveform /= np.max(np.abs(waveform))
    waveform = torch.from_numpy(waveform).float()

    torchaudio.save(output_path, waveform, sample_rate,
    encoding="PCM_S", bits_per_sample=bit_rate)
    
    print(f"Saved {output_path}")


def main(duration: float, sample_rate: int, bit_rate:int, audiodir: str):

    # Generate the arrays
    sweep, inverse_filter, reference = generate_reference(duration, sample_rate)
    single_impulse = impulse(sample_rate, duration, decibels=-18)

    directory = "audio/signals"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save as .wav files
    save_audio(audiodir, "signals/sweep_tone", sample_rate, bit_rate, sweep)
    save_audio(audiodir, "signals/inverse_filter", sample_rate, bit_rate, inverse_filter)
    save_audio(audiodir, "signals/ir_reference", sample_rate, bit_rate, reference)
    save_audio(audiodir, "signals/single_impulse", sample_rate, bit_rate, single_impulse)

    # Plot them
    fig, ax = plt.subplots(3, 1, figsize=(15,7))
    plot_data(get_time_stamps_np(len(sweep), sample_rate), sweep, ax[0], "Sweep Tone", "Time [s]", "Amplitude")
    plot_data(get_time_stamps_np(len(inverse_filter), sample_rate), inverse_filter, ax[1], "Inverse Filter", "Time [s]", "Amplitude")
    plot_data(get_time_stamps_np(len(reference), sample_rate), reference, ax[2], "Reference IR", "Time [s]", "Amplitude")
    fig.suptitle(f"Impulse Response δ(t)")
    save_plot(fig, "Generator_Reference")


if __name__ == "__main__":

    args = parse_args()

    main(args.duration, args.sample_rate, args.bit_rate, args.audiodir)


    # proc_sweep, _ = torchaudio.load('audio/gcn-250_20230824-011632.wav')
    # inv_filt, _ = torchaudio.load('audio/gen/invfilt_gen.wav')
    # ir_result = np.convolve(proc_sweep.squeeze().numpy(), inv_filt.squeeze().numpy())
    # save_audio(args.audiodir, "gcn-250-IR", args.sample_rate, ir_result)