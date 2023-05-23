from pathlib import Path
from scipy.io import wavfile
import os
import h5py
from config import parser

args = parser.parse_args()

def get_length(file_path):
    if file_path.endswith(('.h5')):
        with h5py.File(file_path, 'r') as f:
            # Get the total length of all samples in the file
            sample_rate = args.sr  # Replace with your actual sample rate
            total_length_seconds = 0
            total_length_samples = 0

            for dataset_key in f.keys():
                audio_data = f[dataset_key][:]
                num_samples = audio_data.shape[0]
                audio_length_samples = audio_data.shape[1]
                audio_length_seconds = num_samples * audio_length_samples / sample_rate
                total_length_seconds += audio_length_seconds
                total_length_samples += num_samples * audio_length_samples
    else:
        # Load audio file with scipy
        audio_info = wavfile.info(file_path)

        # Calculate audio length in seconds and samples
        total_length_seconds = audio_info.duration
        total_length_samples = audio_info.frames

    return total_length_seconds, total_length_samples


def main(folder):
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)

        # Check if the file is an audio file by looking at its extension
        if file_path.endswith(('.wav', '.flac', '.ogg', '.aiff', '.caf', '.h5')):
            total_length_seconds, total_length_samples = get_length(file_path)
            print(f'File: {file_name}, Total Length: {total_length_seconds:.2f} seconds, {total_length_samples} samples')


if __name__ == "__main__":
    args = parser.parse_args()

    target_dir = Path(args.root_dir) / args.target_dir

    if not target_dir.is_dir():
        print("The target is not a directory")
        raise SystemExit(1)
  
    main(target_dir)
