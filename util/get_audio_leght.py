import soundfile as sf
import os
import argparse
import h5py

def get_audio_length(file_path):
    if file_path.endswith(('.h5')):
        with h5py.File(file_path, 'r') as f:
            # Get the total length of all samples in the file
            sample_rate = 16000  # Replace with your actual sample rate
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
        # Load audio file with SoundFile
        audio_info = sf.info(file_path)

        # Calculate audio length in seconds and samples
        total_length_seconds = audio_info.duration
        total_length_samples = audio_info.frames

    return total_length_seconds, total_length_samples

def main(audio_directory):
    # Iterate over all audio files in the directory
    for file_name in os.listdir(audio_directory):
        file_path = os.path.join(audio_directory, file_name)

        # Check if the file is an audio file by looking at its extension
        if file_path.endswith(('.wav', '.flac', '.ogg', '.aiff', '.caf', '.h5')):
            total_length_seconds, total_length_samples = get_audio_length(file_path)
            print(f'Audio file: {file_name}, Total Length: {total_length_seconds:.2f} seconds, {total_length_samples} samples')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure the length of audio files in a directory")
    parser.add_argument('audio_directory', type=str, help="Path to the directory containing audio files")

    args = parser.parse_args()

    main(args.audio_directory)

