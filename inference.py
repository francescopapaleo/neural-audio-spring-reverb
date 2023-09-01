import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from src.models.helpers import select_device, load_model_checkpoint
from src.default_args import parse_args

def make_inference(input: torch.Tensor, sample_rate, model, device, mix) -> torch.Tensor:   
    """
    Make inference with the model on the input tensor
    =================================================

    Parameters
    ----------
    input : torch.Tensor
        Input signal shape: [channels, samples]
    sample_rate : int
        Sample rate of the input signal
    model : torch.nn.Module
        Pre-trained model
    device : torch.device
        Device to run the model on
    mix : int
        Mix level of the processed signal

    Returns
    -------
    torch.Tensor
        Processed signal with the same shape as the input signal [channels, samples]
    """
    input = torch.tensor(input, dtype=torch.float32).to(device)
    x = input.reshape(1, 1, -1)         # Add the batch dimension
    
    model.eval()
    with torch.no_grad():
        start_time = datetime.now()

        # Process audio with the pre-trained model
        y_wet = model(x)
        
        end_time = datetime.now()
        duration = end_time - start_time
        num_samples = x.size(-1)
        length_in_seconds = num_samples / sample_rate
        rtf = duration.total_seconds() / length_in_seconds
        print(f"RTF: {rtf}")

    # Normalize
    y_wet /= y_wet.abs().max()

    # High-pass filter
    y_wet = torchaudio.functional.highpass_biquad(y_wet, sample_rate, 5)

    y_wet = y_wet.view(1, -1)
    
    return y_wet


def main():
    args = parse_args()
    device = select_device(args.device)

    model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params = load_model_checkpoint(device, args.checkpoint, args)

    waveform, sr, = torchaudio.load(args.input)

    print(f'input waveform: {waveform.shape}')

    y_hat = make_inference(waveform, args.sample_rate, model, device, args.mix)

    # Save the output using torchaudio
    print(f'output waveform: {y_hat.shape}')

    # Output file path
    filename = f"{model_name}.wav"
    output_file_path = Path(args.audiodir) / f'proc/{filename}'
    torchaudio.save(str(output_file_path), y_hat, sample_rate=args.sample_rate, channels_first=True, bits_per_sample=24)
    print(f"Saved processed file to {output_file_path}")


if __name__ == "__main__":
    
    
    main()