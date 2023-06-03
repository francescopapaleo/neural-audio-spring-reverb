from scipy.io import wavfile
import os
import h5py
from argparse import ArgumentParser


def get_length(file_path):
    total_number_of_frames = 0  # Initialize total_number_of_frames
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
                total_number_of_frames += num_samples  # Increase the total number of frames

            return total_length_seconds, total_length_samples, total_number_of_frames

    else:
        # Load audio file with scipy
        audio_info = wavfile.info(file_path)

        # Calculate audio length in seconds and samples
        total_length_seconds = audio_info.duration
        total_length_samples = audio_info.frames
        total_number_of_frames = total_length_samples  # Set total_number_of_frames to total_length_samples for non-h5 files

        return total_length_seconds, total_length_samples, total_number_of_frames


def main(target_folder):
    file_list = os.listdir(target_folder)
    print(f"Files in folder: {file_list}")
    print("")

    for file_name in file_list:
        file_path = os.path.join(target_folder, file_name)
        print(f"Checking file: {file_path}")
        print("")

        # Check if the file is an audio file by looking at its extension
        if file_path.endswith(('.wav', '.flac', '.ogg', '.aiff', '.caf', '.h5')):
            print(f"Processing file: {file_path}")
            total_length_seconds, total_length_samples, total_number_of_samples = get_length(file_path)
            print(f'File: {file_name}, Total Length: {total_length_seconds:.2f} seconds, {total_length_samples} samples, {total_number_of_samples} total sounds')
            print("")


if __name__ == "__main__":
    parser = ArgumentParser(description="A script to compute the length of audio files")
    parser.add_argument('--target_folder', type=str, required=True, help="Path to the folder containing audio files")
    parser.add_argument('--sr', type=int, default=16000, help="Sample rate of the audio files")
    args = parser.parse_args()

    main(args.target_folder)