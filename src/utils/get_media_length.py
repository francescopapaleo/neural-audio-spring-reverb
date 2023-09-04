from scipy.io import wavfile
import h5py
import os


def get_media_length(file_path):
    file_list = os.listdir(file_path)
    print(f"Files in directory: {file_path}")
    print("")

    for file_name in file_list:
        single_file_path = os.path.join(file_path, file_name)
        print(f"Processing file: {single_file_path}")

        if single_file_path.endswith(('.h5')):
            with h5py.File(single_file_path, 'r') as f:
                sample_rate = sample_rate
                total_length_samples = 0
                total_number_of_frames = 0

                for dataset_key in f.keys():
                    audio_data = f[dataset_key][:]
                    num_samples = audio_data.shape[0]
                    audio_length_samples = audio_data.shape[1]
                    total_length_samples += audio_length_samples
                    total_number_of_frames += num_samples

                total_length_seconds = total_length_samples / sample_rate
                minutes, seconds = divmod(total_length_seconds, 60)
                print(f"Length in {single_file_path}: {int(minutes)} minutes, {seconds:.2f} seconds")
                print(f"Number of elements in {single_file_path}: {total_number_of_frames}")

        elif single_file_path.endswith(('.wav', '.wave')):
            sample_rate, audio_data = wavfile.read(single_file_path)
            total_length_samples = audio_data.shape[0]
            total_length_seconds = total_length_samples / sample_rate
            total_number_of_frames = total_length_samples
            minutes, seconds = divmod(total_length_seconds, 60)
            print(f"Length in {single_file_path}: {int(minutes)} minutes, {seconds:.2f} seconds")
            print(f"Number of elements in {single_file_path}: {total_number_of_frames}")
            
        else:
            print(f"Skipping {single_file_path} (unsupported file type)")

        print("")
