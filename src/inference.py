import torch
import torchaudio
import os
from random import sample
from pathlib import Path
from datetime import datetime

from src.utils.checkpoints import load_model_checkpoint

def make_inference(args) -> torch.Tensor:   
    """
    Make inference with the model on the input tensor
    =================================================

    Parameters
    ----------
    input : torch.Tensor
        Input signal shape: [channels, samples]

    Returns
    -------
    torch.Tensor
        Processed signal with the same shape as the input signal [channels, samples]
    """
    # Load the model
    model, _, _, config, rf, params = load_model_checkpoint(args)

    if isinstance(args.input, str):  # Check if input is a string (file path)
        input, sample_rate = torchaudio.load(args.input)
    else:
        input = torch.tensor(args.input, dtype=torch.float32)
    
    # Add the batch dimension
    input = input.reshape(1, 1, -1).to(args.device)
    
    c = torch.tensor([0., 0.], device=args.device).view(1,1,-1)

    model.eval()
    with torch.no_grad():
        start_time = datetime.now()

        # Process audio with the pre-trained model
        pred = model(input, c)
        
        end_time = datetime.now()
        duration = end_time - start_time
        num_samples = input.size(-1)
        length_in_seconds = num_samples / config['sample_rate']
        rtf = duration.total_seconds() / length_in_seconds
        print(f"RTF: {rtf}")

    # Normalize
    pred /= pred.abs().max()
    # High-pass filter
    pred = torchaudio.functional.highpass_biquad(pred, config['sample_rate'], 5)
    # Remove batch dimension
    pred = pred.view(1, -1)

    pred = pred.view(-1).unsqueeze(0).cpu()
    pred /= torch.max(torch.abs(pred))

    file_name = Path(args.input).stem
    sr_tag = str(int(config['sample_rate'] / 1000)) + 'k'

    os.makedirs(f"{args.audio_dir}/processed", exist_ok=True)
    save_out = f"{args.audio_dir}/processed/{file_name}*{config['name']}.wav"
    
    torchaudio.save(save_out, pred, sample_rate=config['sample_rate'])

    return pred