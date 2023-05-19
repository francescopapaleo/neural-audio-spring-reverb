import torch
import numpy as np


def sine(sample_rate: int, duration: float, amplitude: float, frequency: float = 440.0) -> np.ndarray:
    N = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * np.sin(2.0 * np.pi * frequency * N)

def sweep_tone(
    sample_rate: int,
    duration: float,
    amplitude: float,
    f0: float = 20,
    f1: float = 20000,
    inverse: bool = False,
) -> np.ndarray:

    R = np.log(f1 / f0)
    t = np.arange(0, duration, 1.0 / sample_rate)
    output = np.sin((2.0 * np.pi * f0 * duration / R) * (np.exp(t * R / duration) - 1))
    if inverse:
        k = np.exp(t * R / duration)
        output = output[::-1] / k
    return amplitude * output

def generate_impulse(duration, sample_rate):
    amplitude = 1.0 # Choose an appropriate value for amplitude
    sine_wave = sine(sample_rate, duration, amplitude)
    sweep = sweep_tone(sample_rate, duration, amplitude)

    impulse = np.convolve(sine_wave, sweep, mode='same')

    return torch.from_numpy(impulse).float().unsqueeze(0)  # convert to tensor and add batch dimension

def feed_model_with_impulse(model, impulse):
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():  # disable gradient computation
        output = model(impulse)  # feed the impulse to the model

    return output

