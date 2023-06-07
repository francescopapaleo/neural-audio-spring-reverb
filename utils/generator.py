# generator.py

import numpy as np
import soundfile as sf
from pathlib import Path
from argparse import ArgumentParser

from matplotlib import pyplot as plt

def sine(sample_rate: int, 
         duration: float, 
         amplitude: float, 
         frequency: float = 440.0) -> np.ndarray:
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
    return amplitude * np.sin(2.0 * np.pi * frequency * N)


def sweep_tone(sample_rate: int, 
               duration: float, 
               amplitude: float, 
               f0: float = 20, 
               f1: float = 20000, 
               inverse: bool = False) -> np.ndarray:
    '''
    Generate a sweep tone
    
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
    return amplitude * output


def generate_reference(duration: float, sample_rate: int):
    '''
    Generate and save reference tone files

    Arguments:
    ----------
        duration (float): Duration of the tone.
        sample_rate (int): Sample rate.
    '''
    decibels = -18
    amplitude = 10 ** (decibels / 20)
    f0 = 5
    f1 = sample_rate / 2

    # Generate the sweep tone and inverse filter
    sweep = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1)
    inverse_filter = sweep_tone(sample_rate, duration, amplitude, f0=f0, f1=f1, inverse=True)
    
    # Convolves the sweep tone with the inverse filter in order to obtain the impulse response x(t)
    N = len(sweep)
    reference = np.convolve(inverse_filter, sweep)

    # Save as .wav files
    sweep_output_path = Path('./data/raw') / "sweep.wav"
    sf.write(sweep_output_path, sweep, sample_rate)
    
    inverse_output_path = Path('./data/raw')  / "inverse_filter.wav"
    sf.write(inverse_output_path, inverse_filter, sample_rate)

    reference_output_path = Path('./data/raw')  / "reference.wav"
    sf.write(reference_output_path, reference, sample_rate)

    return sweep, inverse_filter, reference

def plot_generator(sweep_filt: np.ndarray, 
                   inverse_filter: np.ndarray,
                   measured: np.ndarray, 
                   duration: float,
                   sample_rate: int,
                   file_name: str):
    '''
    Plot and save the sweep tone, the inverse filter, and the measured impulse response
    
    Arguments:
    ----------
        sweep_filt (np.ndarray): The sweep tone.
        inverse_filter (np.ndarray): The inverse filter.
        measured (np.ndarray): The measured impulse response.
        sample_rate (int): The sampling frequency.
        duration (float): Duration of the sweep tone.
        file_name (str): Filename to save the plot.
    '''
    time_stamps = np.arange(0, duration, 1 / sample_rate)
    fig, ax = plt.subplots(3, 1, figsize=(15,7))
    
    ax[0].plot(time_stamps, sweep_filt)
    ax[0].set_xlim([0, time_stamps[-1]])
    ax[0].set_title("Sweep Tone")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Amplitude")
    
    ax[1].plot(time_stamps, inverse_filter)
    ax[1].set_xlim([0, time_stamps[-1]])
    ax[1].set_title("Inverse Filter")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Amplitude")
    
    time_stamps = np.arange(0, len(measured)/ sample_rate, 1/ sample_rate)
    ax[2].plot(time_stamps, measured)
    ax[2].set_xlim([0, time_stamps[-1]])
    ax[2].set_title("Impulse Response")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(Path('./data/plots') / file_name)
    plt.close(fig)
    print("Saved signal plot to: ", Path('./data/plots') / file_name)

def main(args):
    '''
    Main function to generate and plot tones
    
    Arguments:
    ----------
        args: Command-line arguments.
    '''
    # Generate and save the impulse, sine wave, and sweep tone
    sweep, inverse_filter, reference = generate_reference(args.duration, args.sample_rate)

    # Plot them
    plot_generator(sweep, inverse_filter, reference, args.duration, args.sample_rate, "reference.png")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample_rate", type=int, default=16000, help="sample rate")
    parser.add_argument("--duration", type=float, default=5.0, help="duration in seconds")
    args = parser.parse_args()

    main(args)