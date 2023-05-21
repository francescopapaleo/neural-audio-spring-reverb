import json
from pathlib import Path
from argparse import ArgumentParser

import h5py


def print_structure(group, indent=""):
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            print(f'{indent}- Group: {key}')
            print_structure(value, indent + '  ')
        else:
            print(f'{indent}- Key: {key}, Shape: {value.shape}, Dtype: {value.dtype}')
            if len(value.attrs) > 0:
                print(f'{indent}  Attributes:')
                for attr_key, attr_value in value.attrs.items():
                    print(f'{indent}    {attr_key}: {attr_value}')
            
            sample_rate = value.attrs.get('sample_rate', 16000)
            num_samples = value.shape[0]
            audio_length_samples = value.shape[1]
            audio_length_seconds = num_samples * audio_length_samples / sample_rate
            print(f'{indent}  Total Length: {audio_length_seconds:.2f} seconds, {num_samples * audio_length_samples} samples')

            # Print information about the first element
            first_element = value[0, :, :]
            print(f"{indent}  First Element:")
            print(first_element)


def main(directory_path):
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        if file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                print(f'Inspecting file: {file_name}')
                print_structure(f)
                print('')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the structure of HDF5 files in a directory")
    parser.add_argument('directory_path', type=str, help="Path to the directory containing HDF5 files")

    args = parser.parse_args()

    main(args.directory_path)

