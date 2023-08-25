import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from src.helpers import select_device, load_model_checkpoint
from configurations import parse_args

def make_inference(input, sample_rate, model, device, mix):   
    
    # Add the batch dimension if it's missing
    input = input.reshape(1, 1, -1)
    input = torch.from_numpy(input).float()

    input = input.to(device)
    
    model.eval()
    with torch.no_grad():
        start_time = datetime.now()

        # Process audio with the pre-trained model
        y_wet = model(input)
        
        end_time = datetime.now()
        duration = end_time - start_time
        num_samples = input.size(1)
        length_in_seconds = num_samples / sample_rate
        rtf = duration.total_seconds() / length_in_seconds
        print(f"RTF: {rtf}")

    # Normalize for safe measure
    y_wet /= y_wet.abs().max()
    
    # Apply mixing
    mix = mix / 100.0
    y_hat = mix * y_wet + (1 - mix) * input

    # Normalize output
    y_hat /= y_hat.abs().max().item()

    return y_hat


def main():
    args = parse_args()
    device = select_device(args.device)

    model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params = load_model_checkpoint(device, args.checkpoint, args)

    waveform, sr, = torchaudio.load(args.input)

    target_length = 240000
    if waveform.size(1) < target_length:
        padding = target_length - waveform.size(1)
        waveform = F.pad(waveform, (0, padding))
    
    waveform = waveform.numpy()
    print(f'input waveform: {waveform.shape}')

    y_hat = make_inference(waveform, args.sample_rate, model, device, args.mix)

    # Create formatted filename
    now = datetime.now()
    filename = f"{model_name}.wav"

    # Output file path
    output_file_path = Path(args.audiodir) / f'proc/{filename}'

    # Save the output using torchaudio
    y_hat = y_hat.squeeze(0).cpu()
    # y_hat = y_hat.cpu()

    torchaudio.save(str(output_file_path), y_hat, sample_rate=args.sample_rate, channels_first=True, bits_per_sample=24)
    print(f"Saved processed file to {output_file_path}")


if __name__ == "__main__":
    main()