import csv
import os
import time

from configurations import parse_args
from src.helpers import load_audio, select_device, load_model_checkpoint
from inference import make_inference
from datetime import datetime
from pathlib import Path
import torch
import torchaudio

def main():
    args = parse_args()

    device = select_device(args.device)

    # New code to iterate over models in a directory
    model_directory = Path(args.checkpoint_path)
    model_paths = [p for p in model_directory.iterdir() if p.is_file() and p.suffix == '.pt']
    saving_path = Path(args.audiodir)

    x_p, fs_x, input_name = load_audio(args.input, args.sample_rate)

    # Open the CSV file for writing
    with open('model_inference_times.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['Model Name', 'Inference Time (s)'])

        for model_path in model_paths:
            start_time = time.time()  # Record start time for inference

            model, model_name, hparams = load_model_checkpoint(device, str(model_path))

            y_hat = make_inference(x_p, fs_x, model, device, args.max_length, args.stereo, args.tail, args.width, args.c0, args.c1, args.gain_dB, args.mix)

            # Record end time for inference and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            print(f"Inference Time for {model_path.name}: {duration:.2f} seconds")

            # Write the model name and inference time to the CSV file
            writer.writerow([model_path.name, duration])

            # Create formatted filename
            now = datetime.now()
            filename = f"{input_name}_{hparams['conf_name']}.wav"  # Use the model filename without extension

            # Output file path
            output_file_path = Path(args.audiodir) / filename

            # Save the output using torchaudio
            y_hat = y_hat.cpu()
            torchaudio.save(str(output_file_path), y_hat, sample_rate=fs_x, channels_first=True, bits_per_sample=16)
            print(f"Saved processed file to {output_file_path}")


if __name__ == "__main__":
    main()

