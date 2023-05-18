import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomH5Dataset(Dataset):
    def __init__(self, input_data_list, output_data_list):
        self.input_data_list = input_data_list
        self.output_data_list = output_data_list

    def __len__(self):
        return len(self.input_data_list)

    def __getitem__(self, idx):
        input_data = self.input_data_list[idx]
        output_data = self.output_data_list[idx]
        
        # Reshape the input and output data
        input_data = input_data.squeeze(-1)
        output_data = output_data.squeeze(-1)

        return input_data, output_data

def load_data(h5_data_path):
    train_dry_data_list = []
    train_wet_data_list = []
    val_dry_data_list = []
    val_wet_data_list = []

    for file_name in os.listdir(h5_data_path):
        if file_name.endswith('.h5'):
            with h5py.File(os.path.join(h5_data_path, file_name), 'r') as f:
                for dataset_name in f.keys():
                    data = torch.from_numpy(np.array(f[dataset_name]))

                    if 'train' in file_name:
                        if 'dry' in file_name:
                            train_dry_data_list.append(data)
                        elif 'wet' in file_name:
                            train_wet_data_list.append(data)
                    elif 'val' in file_name or 'validation' in file_name:
                        if 'dry' in file_name:
                            val_dry_data_list.append(data)
                        elif 'wet' in file_name:
                            val_wet_data_list.append(data)

    return train_dry_data_list, train_wet_data_list, val_dry_data_list, val_wet_data_list

