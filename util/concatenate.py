import h5py
import torch
from torch.utils.data import Dataset

class ConcatenatedDataset(Dataset):
    def __init__(self, x_file, y_file):
        with h5py.File(x_file, 'r') as h5f:
            self.x_data = h5f['Xtrain_subset'][:]
        
        with h5py.File(y_file, 'r') as h5f:
            self.y_data = h5f['Ytrain_0_subset'][:]
        
        self.length = self.x_data.shape[0]

    def __getitem__(self, index):
        x_sample = torch.tensor(self.x_data[index], dtype=torch.float32)
        y_sample = torch.tensor(self.y_data[index], dtype=torch.float32)

        return x_sample, y_sample

    def __len__(self):
        return self.length

# Instantiate the custom dataset and DataLoader
concatenated_dataset = ConcatenatedDataset('dry_train_subset.h5', 'wet_train_subset.h5')
train_dataloader = DataLoader(concatenated_dataset, batch_size=1, shuffle=True)
