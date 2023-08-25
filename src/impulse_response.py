""" Impulse response and transfer function generator.
"""

import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.signals import generate_reference
from src.plotter import plot_impulse_response
from src.helpers import load_model_checkpoint
from inference import make_inference
from configurations import parse_args
from mpl_toolkits.mplot3d import Axes3D

def measure_impulse_response(checkpoint, sample_rate, device, duration, args):
    
    model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params = load_model_checkpoint(device, checkpoint, args)

    # Generate the reference signals
    sweep, inverse_filter, reference = generate_reference(duration, sample_rate) 
    sweep = sweep.reshape(1, -1)

    # Make inference with the model on the sweep tone
    sweep_output = make_inference(
        sweep, sample_rate, model, device, 100)

    # Perform the operation
    sweep_output = sweep_output[0].cpu().numpy()
    sweep_output = sweep_output.squeeze()
    # print(f"sweep_output:{sweep_output.shape} || inverse_filter:{inverse_filter.shape}")

    # Convolve the sweep tone with the inverse filter
    measured = np.convolve(sweep_output, inverse_filter)
    
    # measured_index = np.argmax(np.abs(measured))
    # print(f"measured_index:{measured_index}")

    # Save and plot the measured impulse response
    save_as = f"{Path(checkpoint).stem}_IR.wav"
    wavfile.write(f"results/measured_IR/{save_as}", sample_rate, measured.astype(np.float32))
    print(f"Saved measured impulse response to {save_as}")

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
    for ax in axs:
        ax.grid(True)
        ax.set_xlabel('Time [sec]')
    
    model_label = hparams['conf_name']
    axs[0].set_title(f'Model: {model_label} Impulse Response (IR)')
    time_axis = np.arange(len(measured)) / sample_rate
    axs[0].plot(time_axis, measured)
    axs[0].set_ylim([-1, 1])
    
    axs[1].specgram(measured, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='hot', scale='dB')
    
    plt.tight_layout()
    plt.savefig(f"results/measured_IR/{Path(checkpoint).stem}_IR.png")
    print(f"Saved measured impulse response plot to {Path(checkpoint).stem}_IR.png")
    plt.show()
        
    return sweep_output, measured


if __name__ == "__main__":
    args = parse_args()

    print('Measure the impulse response of a trained model')

    measure_impulse_response(args.checkpoint, args.sample_rate, args.device, args.duration, args)