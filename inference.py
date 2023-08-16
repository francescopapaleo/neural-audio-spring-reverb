# inference.py

import torch
import torchaudio
import torchaudio.functional as F
from pathlib import Path
from datetime import datetime
import time

from src.helpers import load_audio, select_device, load_model_checkpoint
from configurations import parse_args

torch.manual_seed(42)

def make_inference(x_p, sample_rate, model, device, c0: float, c1: float, mix: float, ) -> torch.Tensor:   
    x_p_mono = x_p.mean(dim=0, keepdim=True)
    x_p_mono = x_p_mono.unsqueeze(0)  # Add batch dimension
    print(f"Shape of mono audio: {x_p_mono.shape}")

    c = torch.tensor([0.0, 0.0]).view(1,1,-1)  

    x_p = x_p_mono.to(device)
    c = c.to(device)

    # Pad the input signal 
    front_pad = 256 - 1
    back_pad = front_pad
    x_p_pad = torch.nn.functional.pad(x_p, (front_pad, back_pad))

    # Process audio with the pre-trained model
    model.eval()
    with torch.no_grad():
        
        y_wet = model(x_p_pad, c)
        y_wet_hp = F.highpass_biquad(y_wet.view(-1), sample_rate, 20.0)
        y_wet_lp = F.lowpass_biquad(y_wet_hp.view(-1), sample_rate, 20000.0)
    
    # Normalize for safe measure
    y_wet /= y_wet_lp.abs().max()
    x_dry = x_p_pad.abs().max()
    
    y_wet = y_wet.to(device)
    x_dry = x_dry.to(device)

    # Mix
    mix = mix / 100.0
    y_hat = (mix * y_wet) + ((1 - mix) * x_dry)

    # # Remove transient
    # y_hat = y_hat[..., 8192:]
    y_hat /= y_hat.abs().max().item()

    return y_hat

    return y_wet


def main():
    args = parse_args()

    device = select_device(args.device)

    model, model_name, hparams = load_model_checkpoint(device, args.checkpoint, args)

    x_p, fs_x, = torchaudio.load(args.input)

    y_hat = make_inference(x_p, args.sample_rate, model, device, args.c0, args.c1, args.mix)

    # Create formatted filename
    now = datetime.now()
    filename = f"{model_name}.wav"

    # Output file path
    output_file_path = Path(args.audiodir) / filename

    # Save the output using torchaudio
    y_hat = y_hat.squeeze(0)
    y_hat = y_hat.cpu()
    print(y_hat.shape)
    torchaudio.save(str(output_file_path), y_hat, sample_rate=args.sample_rate, channels_first=True, bits_per_sample=24)
    print(f"Saved processed file to {output_file_path}")


if __name__ == "__main__":
    main()