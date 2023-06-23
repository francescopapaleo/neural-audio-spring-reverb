# inference.py

import torch
import torchaudio
import torchaudio.functional as F
from pathlib import Path
from datetime import datetime

from utils.helpers import load_audio, select_device, load_model_checkpoint
from config import parse_args

torch.manual_seed(42)


def make_inference(x_p, fs_x, model, device, max_length: float, stereo: bool, tail: float, width: float, c0: float, c1: float, gain_dB: float, mix: float) -> torch.Tensor:
    
    x_p = x_p.to(device)
    
    rf = model.compute_receptive_field()
    
    chs = x_p.shape[0]
    # If mono and stereo requested
    if chs == 1 and stereo:
        x_p = x_p.repeat(2, 1)
        chs = 2

    # Pad the input signal 
    front_pad = rf - 1
    back_pad = 0 if not tail else front_pad
    x_p_pad = torch.nn.functional.pad(x_p, (front_pad, back_pad))

    # Compute linear gain
    gain_ln = 10 ** (gain_dB / 20.0)

    # Process audio with the pre-trained model
    with torch.no_grad():
        y_wet = torch.zeros((chs, x_p_pad.shape[1]))

        for n in range(chs):
            if n == 0:
                factor = (width * 5e-3)
            elif n == 1:
                factor = -(width * 5e-3)
            c = torch.tensor([float(c0 + factor), float(c1 + factor)]).view(1, 1, -1).to(device)
        
            y_wet_ch = model(gain_ln * x_p_pad[n, :].view(1, 1, -1), c)

            y_wet_ch = F.highpass_biquad(y_wet_ch.view(-1), fs_x, 20.0)
            y_wet_ch = F.lowpass_biquad(y_wet_ch.view(-1), fs_x, 20000.0)

            y_wet[n, :] = y_wet_ch

    x_dry = x_p_pad.to(device)

    # Normalize each first
    y_wet /= y_wet.abs().max()
    x_dry /= x_p_pad.abs().max()

    y_wet = y_wet.to(device)
    x_dry = x_dry.to(device)

    # Mix
    mix = mix / 100.0
    y_hat = (mix * y_wet) + ((1 - mix) * x_dry)

    # # Remove transient
    y_hat = y_hat[..., 8192:]
    y_hat /= y_hat.abs().max().item()

    return y_hat


def main():
    args = parse_args()

    device = select_device(args.device)

    model, model_name, hparams = load_model_checkpoint(device, args.checkpoint_path)

    x_p, fs_x, input_name = load_audio(args.input, args.sample_rate)

    y_hat = make_inference(x_p, fs_x, model, device, args.max_length, args.stereo, args.tail, args.width, args.c0, args.c1, args.gain_dB, args.mix)

    # Create formatted filename
    now = datetime.now()
    filename = f"{input_name}_{model_name}.wav"

    # Output file path
    output_file_path = Path(args.audiodir) / filename

    # Save the output using torchaudio
    y_hat = y_hat.cpu()
    torchaudio.save(str(output_file_path), y_hat, sample_rate=args.sample_rate, channels_first=True, bits_per_sample=16)
    print(f"Saved processed file to {output_file_path}")


if __name__ == "__main__":
    main()