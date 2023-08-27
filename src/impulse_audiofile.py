
import torch
import torchaudio
import numpy as np
import scipy.signal as signal
from pathlib import Path
import matplotlib.pyplot as plt

from configurations import parse_args
from src.plotter import plot_waterfall    

def main(input_path, args):
    """
    Extract the impulse response of an input file
    =============================================
    1. Load the input file
    2. Load the inverse filter
    3. Convolve the input file with the inverse filter
    4. Normalize the impulse response
    5. Plot the spectrogram
    6. Save the impulse response as a .wav file

    """

    input, sample_rate = torchaudio.load(input_path)
    input_np = input.view(-1).squeeze(0).cpu().numpy()
    
    inverse_filter, _ = torchaudio.load("audio/signals/inverse_filter.wav")
    inverse_filter = inverse_filter.view(-1).squeeze(0).cpu().numpy()

    print(f"input:{input_np.shape} || inverse_filter:{inverse_filter.shape}")

    # Convolve the sweep tone with the inverse filter
    impulse_response = np.convolve(input_np, inverse_filter)
    print(impulse_response.shape)

    # Normalize the impulse response
    impulse_response /= np.max(np.abs(impulse_response))
        
    # Only create a single subplot for the spectrogram
    fig, ax = plt.subplots(figsize=(10, 8))

    file_name = Path(input_path).stem

    # Plot the spectrogram
    cax = ax.specgram(impulse_response, NFFT=512, Fs=sample_rate, noverlap=256, cmap='hot', scale='dB', mode='magnitude', vmin=-100, vmax=0)
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(f'{file_name} IR')
    ax.grid(True)

    # Add the colorbar
    cbar = fig.colorbar(mappable=cax[3], ax=ax, format="%+2.0f dB")
    cbar.set_label('Intensity [dB]')

    plt.tight_layout()
    plt.savefig(f"results/measured_IR/{file_name}_IR.png")
    print(f"Saved spectrogram plot to {file_name}_IR.png")

    ir_tensor = torch.from_numpy(impulse_response).unsqueeze(0).float()
    save_as = Path(f"results/measured_IR/{file_name}_IR.wav")
    torchaudio.save(save_as, ir_tensor, sample_rate, bits_per_sample=args.bit_depth)
    print(f"Saved measured impulse response to {save_as}, sample rate: {sample_rate}, bit depth: {args.bit_depth}")

    # Plot the waterfall spectrogram
    plot_waterfall(impulse_response, file_name, sample_rate, stride=10)

if __name__ == "__main__":
    args = parse_args()

    main(args.input, args)