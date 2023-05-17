import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

H5_DATA = '/homedtic/fpapaleo/smc-spring-reverb/dataset'

# Instantiate the dataset

class CustomH5Dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
train_data_list = []
val_data_list = []

for file_name in os.listdir(H5_DATA):
    if file_name.endswith('.h5'):
        with h5py.File(os.path.join(H5_DATA, file_name), 'r') as f:
            for dataset_name in f.keys():
                data = torch.from_numpy(np.array(f[dataset_name]))

                if 'train' in file_name:
                    train_data_list.append(data)
                elif 'val' in file_name or 'validation' in file_name:
                    val_data_list.append(data)

train_dataset = CustomH5Dataset(train_data_list)
val_dataset = CustomH5Dataset(val_data_list)

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Iterate over the train_dataloader
for batch_idx, data in enumerate(train_dataloader):
    print(f"Batch {batch_idx}:")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print()

    # You can break the loop after the first iteration to see just one batch
    break

# Iterate over the val_dataloader
for batch_idx, data in enumerate(val_dataloader):
    print(f"Batch {batch_idx}:")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print()

    # You can break the loop after the first iteration to see just one batch
    break