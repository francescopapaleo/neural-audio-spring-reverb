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
    impulse_response = signal.fftconvolve(sweep_output, inverse_filter, mode='same')
    
    # measured_index = np.argmax(np.abs(measured))
    # print(f"measured_index:{measured_index}")

    # Save and plot the measured impulse response
    save_as = f"{Path(checkpoint).stem}_IR.wav"
    wavfile.write(f"results/measured_IR/{save_as}", sample_rate, impulse_response.astype(np.float32))
    print(f"Saved measured impulse response to {save_as}")

    # Only create a single subplot for the spectrogram
    fig, ax = plt.subplots(figsize=(10, 8))

    model_label = hparams['conf_name']
    ax.set_title(f'Model: {model_label} Spectrogram')
    time_axis = np.arange(len(impulse_response)) / sample_rate

    # Plot the spectrogram
    cax = ax.specgram(impulse_response, NFFT=2048, Fs=sample_rate, noverlap=256, cmap='hot', scale='dB', vmin=-80, vmax=0)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.grid(True)

    # Add the colorbar
    cbar = fig.colorbar(mappable=cax[3], ax=ax, format="%+2.0f dB")
    cbar.set_label('Intensity [dB]')

    plt.tight_layout()
    plt.savefig(f"results/measured_IR/{Path(checkpoint).stem}_IR.png")
    print(f"Saved spectrogram plot to {Path(checkpoint).stem}_IR.png")


    # https://github.com/willfehlmusic/Python_Sketchpads/blob/master/Sketchpad004_LinLogSweep/Sketchpad004_LinLogSweep.py

    log_n = len(sweep_output) # length of the signal
    log_k = np.arange(log_n)
    log_T = log_n/sample_rate
    log_frq = log_k/log_T # two sides frequency range
    log_frq = log_frq[range(log_n//2)] # one side frequency range
    log_Y = np.fft.fft(sweep_output)/log_n # fft computing and normalization
    log_Y = log_Y[range(log_n//2)]

    IR_n = len(impulse_response) # length of the signal
    IR_k = np.arange(IR_n)
    IR_T = IR_n/sample_rate
    IR_frq = IR_k/IR_T # two sides frequency range
    IR_frq = IR_frq[range(IR_n//2)] # one side frequency range
    IR_Y = np.fft.fft(impulse_response)/IR_n # fft computing and normalization
    IR_Y = IR_Y[range(IR_n//2)]

    fig, axes = plt.subplots(2, 1, sharex=False, sharey=False, constrained_layout=True,figsize=(10,5))
    axes[0].plot(impulse_response,'r') # plotting the spectrum
    axes[0].set_title('Time Domain')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_xlim(len(impulse_response)*(0.5-0.005),len(impulse_response)*(0.5+0.005))
    axes[1].semilogx(log_frq, 20*np.log10(abs(IR_Y)),'r') # plotting the spectrum
    axes[1].set_title('Frequency Domain')
    axes[1].set_xlabel('Frequency [Hz]')
    axes[1].set_ylabel('Magnitude [dB]')
    axes[1].set_xlim(20,20000)
    axes[1].set_ylim(-120,0)
    plt.grid()
    plt.savefig('response_of_IR_New.png', bbox_inches="tight")
    plt.show(block=False)


if __name__ == "__main__":
    args = parse_args()

    print('Measure the impulse response of a trained model')

    measure_impulse_response(args.checkpoint, args.sample_rate, args.device, args.duration, args)