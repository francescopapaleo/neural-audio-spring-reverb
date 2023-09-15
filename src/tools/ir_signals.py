import numpy as np
from typing import Tuple

""" 
Signal Generators for Measurements
===================================
This module contains functions to generate the signals used in the measurements:
    - Impulse
    - Sine wave
    - Sweep tone
    - Reference impulse response

Modified version from the original written by Xavier Lizarraga
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

