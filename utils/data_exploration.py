# Description: This file contains the code for data exploration and visualization
from config import parser
from pathlib import Path
from scipy.io import wavfile
from torch.utils.data import Dataset

import os
import h5py
import numpy as np
import random

h5_path = '/Users/francescopapaleo/datasets/plate-spring/spring/'
destination = '/Users/francescopapaleo/git-box/smc-spring-reverb/data/'
seed = 42  # You can choose any seed value
args = parser.parse_args()

random.seed(seed)
np.random.seed(seed)

# Load the data
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

        # Add index to each data item
        self.index = {i: (self.dry_file.name, self.wet_file.name, i) for i in range(len(self.dry_data))}

    def __len__(self):
        # The length should be the number of items in the data
        return self.dry_data.shape[0]
    
    def __getitem__(self, index):
        # Should return a tuple of numpy arrays (x, y )
        return self.dry_data[index], self.wet_data[index]
    
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
        print(f"Shape after concatenation: {concatenated_data.shape}")
        
        # Reshape the data
        # reshape_data = concatenated_data.reshape(1, -1, 1)
        
        # print(f"Shape after reshaping: {reshape_data.shape}")  # Print shape after reshaping
        return concatenated_data
        # return reshape_data
    
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

# Load the dataset
dataset = PlateSpringDataset(h5_path, split='test')
dataset.print_info()

subset = dataset.load_random_subset(10, seed=42)
dataset.print_subset_indexes(subset)

# concatenated_dry = dataset.concatenate_samples()
# concatenated_wet = dataset.concatenate_samples()

# Verify if the dataset contains any metadata
with h5py.File(os.path.join(args.target_dir,'dry_train.h5'), 'r') as f:
    print(f['Xtrain'].attrs.keys())
    print(f['Xtrain'].attrs.values())
    values = list(f['Xtrain'].attrs.values())
    print(values)
