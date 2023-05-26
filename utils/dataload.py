# Description: This file contains the code for the dataset class and the subset generator class.
from pathlib import Path
from config import parser

import h5py
import torch
import numpy as np

args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)


class SpringDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split=None):
        super(SpringDataset, self).__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.seed = torch.seed()

        self.file_list = list(self.root_dir.glob('**/*.h5'))
        self.dry_file = [f for f in self.file_list if 'dry' in f.stem and self.split in f.stem][0]
        self.wet_file = [f for f in self.file_list if 'wet' in f.stem and self.split in f.stem][0]
        self.dry_data = None
        self.wet_data = None
        self.load_data()
        self.index = {i: (self.dry_file.name, self.wet_file.name, i) for i in range(len(self.dry_data))}
        
    def __len__(self):
        # The length should be the number of items in the data
        return len(self.index)
    
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

    
    def print_index_info(self, index):
        """Prints the information about a specific index tuple."""
        print(f"Index: {index}")
        print(f"Index data: {self.index[index]}")

    def load_random_subset(self, size, seed=None):
        """Loads a random subset of the dataset."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        indices = np.random.choice(len(self.dry_data), size=size, replace=False)
        dry_subset = self.dry_data[indices]
        wet_subset = self.wet_data[indices]
        
        return dry_subset, wet_subset, indices
    
    
    def concatenate_samples(self, data):
        print(f"Shape before concatenation: {data.shape}")
        concatenated_data = np.concatenate(data, axis=0)
        reshape_data = concatenated_data.reshape(1, -1)      
        return reshape_data