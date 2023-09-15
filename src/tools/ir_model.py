from logging import config
import numpy as np
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

from src.tools.ir_signals import generate_reference
from src.utils.checkpoints import load_model_checkpoint
from src.inference import make_inference
from src.tools.plotter import plot_ir_spectrogram, plot_waterfall



def measure_model_ir(args):
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

    print('Measure the impulse response of a trained model')

    model, _, _, config, _, _ = load_model_checkpoint(args)

    # Generate the reference signals
    sweep, inverse_filter, _ = generate_reference(args.duration, config['sample_rate']) 
    # sweep:(240000,) || inverse_filter:(240000,)
    sweep = sweep.reshape(1, -1)

    # Make inference with the model on the sweep tone
    # args.input = torch.from_numpy(sweep).float()
    # print(f'input waveform: {input.shape}')
    args.input = sweep
    
    sweep_output = make_inference(args)

    # print(f"sweep_output:{sweep_output.shape} || inverse_filter:{inverse_filter.shape}")
    sweep_output = sweep_output.reshape(-1).numpy()

    # Normalize the sweep tone and the inverse filter
    sweep_output = sweep_output - np.mean(sweep_output)
    sweep_output /= np.max(np.abs(sweep_output))
    
    inverse_filter /= np.max(np.abs(inverse_filter))

    # print("post squeeze")
    # print(f"sweep_output:{sweep_output.shape} || inverse_filter:{inverse_filter.shape}")
    
    # Convolve the sweep tone with the inverse filter
    # impulse_response = np.convolve(sweep_output, inverse_filter, mode='full')
    impulse_response = signal.convolve(sweep_output, inverse_filter, mode='full', method='direct')
    print(f"IR: {impulse_response.shape}")

    # Normalize the impulse response
    impulse_response = impulse_response - np.mean(impulse_response)
    impulse_response /= np.max(np.abs(impulse_response))
        
    # # Only create a single subplot for the spectrogram
    # fig, ax = plt.subplots(figsize=(10, 8))

    # model_label = config['name']
    # ax.set_title(f'Model: {model_label} Spectrogram')
    
    # # Plot the spectrogram
    # cax = ax.specgram(impulse_response, NFFT=512, Fs=config['bit_rate'], noverlap=256, cmap='hot', scale='dB', mode='magnitude', vmin=-100, vmax=0)
    # ax.set_ylabel('Frequency [Hz]')
    # ax.set_xlabel('Time [sec]')
    # ax.grid(True)

    # # Add the colorbar
    # cbar = fig.colorbar(mappable=cax[3], ax=ax, format="%+2.0f dB")
    # cbar.set_label('Intensity [dB]')

    # plt.tight_layout()
    # plt.savefig(f"plots/measured_IR/{Path(args.checkpoint).stem}_IR.png")
    # print(f"Saved spectrogram plot to {Path(args.checkpoint).stem}_IR.png")
    plot_ir_spectrogram(impulse_response, config['sample_rate'], f"Model: {config['name']} Spectrogram", f"plots/measured_IR/{Path(args.checkpoint).stem}_IR.png")

    # Plot the waterfall spectrogram
    file_name = f"{Path(args.checkpoint).stem}_IR_waterfall.png"
    plot_waterfall(impulse_response, file_name, config['sample_rate'], args, stride=10)

    ir_tensor = torch.from_numpy(impulse_response).unsqueeze(0).float()
    
    save_directory = "audio/measured_IR"
    Path(save_directory).mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    save_as = f"{save_directory}/{Path(args.checkpoint).stem}_IR.wav"
    torchaudio.save(save_as, ir_tensor, config['sample_rate'], bits_per_sample=config['bit_rate'])
    print(f"Saved measured impulse response to {save_as}, sample rate: {config['sample_rate']}, bit depth: {config['bit_rate']}")

    