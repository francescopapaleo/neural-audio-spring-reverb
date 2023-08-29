import numpy as np
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt

from src.signals import generate_reference
from src.helpers import load_model_checkpoint
from src.plotter import plot_waterfall
from inference import make_inference
from configurations import parse_args

def measure_impulse_response(checkpoint, sample_rate, bit_rate, device, duration, args):
    """ 
    Impulse response measurement of a trained model
    =========================================================
    1. Generate the analysis signals
    2. Make inference with the model on the sweep tone
    3. Convolve the sweep tone with the inverse filter
    4. Normalize the impulse response
    5. Plot the spectrogram
    6. Save the impulse response as a .wav file
    
    """

    model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params = load_model_checkpoint(device, checkpoint, args)

    # Generate the reference signals
    sweep, inverse_filter, _ = generate_reference(duration, sample_rate) 
    # sweep:(240000,) || inverse_filter:(240000,)
    sweep = sweep.reshape(1, -1)

    # Make inference with the model on the sweep tone
    sweep_output = make_inference(
        sweep, sample_rate, model, device, 100)

    print(f"sweep_output:{sweep_output.shape} || inverse_filter:{inverse_filter.shape}")
    sweep_output = sweep_output.reshape(-1)

    print("post squeeze")
    print(f"sweep_output:{sweep_output.shape} || inverse_filter:{inverse_filter.shape}")


    # Convolve the sweep tone with the inverse filter
    impulse_response = np.convolve(sweep_output, inverse_filter)
    print(impulse_response.shape)

    # Normalize the impulse response
    impulse_response /= np.max(np.abs(impulse_response))
        
    # Only create a single subplot for the spectrogram
    fig, ax = plt.subplots(figsize=(10, 8))

    model_label = hparams['conf_name']
    ax.set_title(f'Model: {model_label} Spectrogram')
    
    # Plot the spectrogram
    cax = ax.specgram(impulse_response, NFFT=512, Fs=sample_rate, noverlap=256, cmap='hot', scale='dB', mode='magnitude', vmin=-100, vmax=0)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.grid(True)

    # Add the colorbar
    cbar = fig.colorbar(mappable=cax[3], ax=ax, format="%+2.0f dB")
    cbar.set_label('Intensity [dB]')

    plt.tight_layout()
    plt.savefig(f"results/measured_IR/{Path(checkpoint).stem}_IR.png")
    print(f"Saved spectrogram plot to {Path(checkpoint).stem}_IR.png")

    # Plot the waterfall spectrogram
    file_name = f"{Path(checkpoint).stem}_IR_waterfall.png"
    plot_waterfall(impulse_response, file_name, sample_rate, stride=10)

    ir_tensor = torch.from_numpy(impulse_response).unsqueeze(0).float()
    
    save_directory = "results/measured_IR/audio"
    Path(save_directory).mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    save_as = f"{save_directory}/{Path(checkpoint).stem}_IR.wav"
    torchaudio.save(save_as, ir_tensor, sample_rate, bits_per_sample=bit_rate)
    print(f"Saved measured impulse response to {save_as}, sample rate: {sample_rate}, bit depth: {bit_rate}")



if __name__ == "__main__":
    args = parse_args()

    print('Measure the impulse response of a trained model')

    measure_impulse_response(args.checkpoint, args.sample_rate, args.bit_rate, args.device, args.duration, args)