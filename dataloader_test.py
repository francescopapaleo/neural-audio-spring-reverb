import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader_subset import CustomH5Dataset, load_data

#### SHOULD BE ######
# [batch_size, num_channels, sequence_length]

H5_DATA = '/homedtic/fpapaleo/smc-spring-reverb/dataset_subset'

# Load the dataset
train_dry_data_list, train_wet_data_list, val_dry_data_list, val_wet_data_list = load_data(H5_DATA)

train_dataset = CustomH5Dataset(train_dry_data_list, train_wet_data_list)
val_dataset = CustomH5Dataset(val_dry_data_list, val_wet_data_list)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

# Iterate over the train_dataloader
for batch_idx, (input_data, output_data) in enumerate(train_dataloader):
    print(f"Batch {batch_idx}:")
    print(f"Input Data shape: {input_data.shape}")
    print(f"Output Data shape: {output_data.shape}")
    print()

    # You can break the loop after the first iteration to see just one batch
    break

# Iterate over the val_dataloader
for batch_idx, (input_data, output_data) in enumerate(val_dataloader):
    print(f"Batch {batch_idx}:")
    print(f"Input Data shape: {input_data.shape}")
    print(f"Output Data shape: {output_data.shape}")
    print()

    # You can break the loop after the first iteration to see just one batch
    break