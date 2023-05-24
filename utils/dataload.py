# Description: This file contains the code for the dataset class and the subset generator class.
from pathlib import Path
from config import parser

import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
import os

args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)


class PlateSpringDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        super(PlateSpringDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.file_list = list(self.root_dir.glob('**/*.h5'))
        self.dry_file = [f for f in self.file_list if 'dry' in f.stem and self.split in f.stem][0]
        self.wet_file = [f for f in self.file_list if 'wet' in f.stem and self.split in f.stem][0]
        self.dry_data = None
        self.wet_data = None
        self.load_data()
        self.index = {i: (self.dry_file.name, self.wet_file.name, i) for i in range(len(self.dry_data))}
        
        # Add index to each data item
        self.index = {i: (self.dry_file.name, self.wet_file.name, i) for i in range(len(self.dry_data))}

    def __len__(self):
        # The length should be the number of items in the data
        return self.dry_data.shape[0]
    
    def __getitem__(self, index):
        # Should return a tuple of numpy arrays (x, y )
        x, y = self.dry_data[index], self.wet_data[index]
        x, y = x.reshape(1, -1), y.reshape(1, -1)    
        return x, y
    
    def load_data(self):
        with h5py.File(self.dry_file, 'r') as f_dry, \
             h5py.File(self.wet_file, 'r') as f_wet:
            dry_key = list(f_dry.keys())[0] 
            wet_key = list(f_wet.keys())[0] 
            self.dry_data = f_dry[dry_key][:]
            self.wet_data = f_wet[wet_key][:]

    def print_info(self):
        print(f"Dry File: {self.dry_file}")
        print(f"Wet File: {self.wet_file}")
        print(f"Total data items: {self.__len__()}")
        print(f"Dry data shape: {self.dry_data.shape}")
        print(f"Wet data shape: {self.wet_data.shape}")
        print(f"(item_n, samples_per_item, channels)")

    def get_index_info(self, index):
        return self.index.get(index)
    
    def load_random_subset(self, size, seed=0):
        np.random.seed(seed)
        indices = np.random.choice(len(self.dry_data), size=size)
        return self.dry_data[indices], self.wet_data[indices], indices

    def concatenate_samples(self, data):
        print(f"Shape before concatenation: {data.shape}")
        concatenated_data = np.concatenate(data, axis=0)
        reshape_data = concatenated_data.reshape(1, -1)      
        return reshape_data
    
    def print_subset_indexes(self, subset):
        """
        Prints the indices selected for the subset.

        Parameters:
        subset: A tuple of numpy arrays representing the chosen indexes.
        """
        dry_data, wet_data, indices = subset
        print(f"Chosen subset items: {indices}")

    def save_subset(self, subset, save_dir, seed):
        """
        Saves the chosen subset of the dataset in a given location.

        Parameters:
        subset: A tuple of numpy arrays representing the chosen subset. 
        (dry_samples_subset, wet_samples_subset, indexes)
        save_dir: A string representing the directory to save the subset.
        seed: An integer representing the seed used for the random subset selection.
        """
        dry_samples_subset, wet_samples_subset, indexes = subset

        dry_file_name = os.path.join(save_dir, f"dry_sub_{seed}.h5")
        wet_file_name = os.path.join(save_dir, f"wet_sub_{seed}.h5")

        with h5py.File(dry_file_name, 'w') as f_dry:
            f_dry.create_dataset("dry_samples_subset", data=dry_samples_subset)
        with h5py.File(wet_file_name, 'w') as f_wet:
            f_wet.create_dataset("wet_samples_subset", data=wet_samples_subset)


# # Main class that retrieves the entire dataset
# class PlateSpring(Dataset):
#     def __init__(self, data_folder):
#         self.data_folder = data_folder

#     def download_data(self):
#         pass

#     def retrieve_data(self):
#         x_train_path = f"{self.data_folder}/dry_train.h5"
#         y_train_path = f"{self.data_folder}/wet_train.h5"
#         x_val_test_path = f"{self.data_folder}/dry_val_test.h5"
#         y_val_test_path = f"{self.data_folder}/wet_val_test.h5"

#         with h5py.File(x_train_path, 'r') as f:
#             x_train = f['Xtrain'][:]

#         with h5py.File(y_train_path, 'r') as f:
#             y_train = f['Ytrain_0'][:]

#         with h5py.File(x_val_test_path, 'r') as f:
#             x_val_test = f['Xvalidation'][:]

#         with h5py.File(y_val_test_path, 'r') as f:
#             y_val_test = f['Yvalidation_0'][:]

#         return x_train, y_train, x_val_test, y_val_test


# class SubsetGenerator(PlateSpring):
#     def __init__(self, data_folder, subset_size):
#         super().__init__(data_folder)
#         self.subset_size = subset_size

#     @staticmethod
#     def generate_idx(data, subset_size):
#         # Generate a list of indices based on the seed
#         np.random.seed(seed)
#         idx = np.random.choice(np.arange(len(data)), size=subset_size, replace=False)
#         return idx

#     def select_random_samples(self, data, selected_idx):
#         # Select the samples corresponding to the provided indices
#         selected_samples = data[selected_idx]
#         return selected_samples

#     def save_subset(self, x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset, save_folder):
#         # Ensure the save folder exists
#         os.makedirs(save_folder, exist_ok=True)

#         with h5py.File(os.path.join(save_folder, 'x_train_subset.h5'), 'w') as f:
#             f.create_dataset('Xtrain_subset', data=x_train_subset)

#         with h5py.File(os.path.join(save_folder, 'y_train_subset.h5'), 'w') as f:
#             f.create_dataset('Ytrain_0_subset', data=y_train_subset)

#         with h5py.File(os.path.join(save_folder, 'x_val_test_subset.h5'), 'w') as f:
#             f.create_dataset('Xvalidation_subset', data=x_val_test_subset)

#         with h5py.File(os.path.join(save_folder, 'y_val_test_subset.h5'), 'w') as f:
#             f.create_dataset('Yvalidation_0_subset', data=y_val_test_subset)
            
#     def retrieve_subset(self):
#         x_train, y_train, x_val_test, y_val_test = self.retrieve_data()

#         # Generate indices for each data subset
#         train_idx = self.generate_idx(x_train, self.subset_size)
#         val_test_idx = self.generate_idx(x_val_test, self.subset_size // 2)  # Assuming you want half the size for val/test
        
#         # Select samples
#         x_train_subset = self.select_random_samples(x_train, train_idx)
#         y_train_subset = self.select_random_samples(y_train, train_idx)
#         x_val_test_subset = self.select_random_samples(x_val_test, val_test_idx)
#         y_val_test_subset = self.select_random_samples(y_val_test, val_test_idx)

#         # Save the subsets to the desired folder
#         self.save_subset(x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset, 'dataset_subset')

#         return x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset

# class SubsetRetriever(Dataset):
#     def __init__(self, data_folder):
#         self.data_folder = data_folder

#     def concatenate_samples(self, data):
#         # Concatenate samples along the first dimension
#         concatenated_data = np.concatenate(data, axis=0)
#         reshape_data = concatenated_data.reshape(1, -1)
#         return reshape_data

#     def retrieve_data(self, concatenate=True):
#         x_train_path = f"{self.data_folder}/dry_train_subset.h5"
#         y_train_path = f"{self.data_folder}/wet_train_subset.h5"
#         x_val_test_path = f"{self.data_folder}/dry_val_test_subset.h5"
#         y_val_test_path = f"{self.data_folder}/wet_val_test_subset.h5"

#         with h5py.File(x_train_path, 'r') as f:
#             x_train_subset = f['Xtrain_subset'][:]

#         with h5py.File(y_train_path, 'r') as f:
#             y_train_subset = f['Ytrain_0_subset'][:]

#         with h5py.File(x_val_test_path, 'r') as f:
#             x_val_test_subset = f['Xvalidation_subset'][:]

#         with h5py.File(y_val_test_path, 'r') as f:
#             y_val_test_subset = f['Yvalidation_0_subset'][:]

#         if concatenate:
#             # Concatenate samples
#             x_train_concatenated = self.concatenate_samples(x_train_subset)
#             y_train_concatenated = self.concatenate_samples(y_train_subset)
#             x_val_test_concatenated = self.concatenate_samples(x_val_test_subset)
#             y_val_test_concatenated = self.concatenate_samples(y_val_test_subset)

#             return x_train_concatenated, y_train_concatenated, x_val_test_concatenated, y_val_test_concatenated
#         else:
#             return x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset

# class AudioDataRetriever(Dataset):
#     def __init__(self, data_folder):
#         self.data_folder = data_folder

#     def retrieve_data(self):
#         audio_files = os.listdir(self.data_folder)
#         audio_data = []

#         for file in audio_files:
#             filepath = os.path.join(self.data_folder, file)
#             waveform, sample_rate = torchaudio.load(filepath)

#             # Normalize to mono (single channel)
#             if waveform.shape[0] > 1:
#                 waveform = waveform.mean(dim=0, keepdim=True)

#             audio_data.append(waveform.numpy())

#         return np.array(audio_data)
    
    