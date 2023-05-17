import torch
import h5py
from torch.utils.data import Dataset

class H5AudioDataset(Dataset):
    def __init__(self, dry_h5_file_path, wet_h5_file_path):
        self.dry_h5_file_path = dry_h5_file_path
        self.wet_h5_file_path = wet_h5_file_path
        
        with h5py.File(self.dry_h5_file_path, 'r') as f:
            self.num_samples = len(f['Xtrain'])
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.dry_h5_file_path, 'r') as f_dry, h5py.File(self.wet_h5_file_path, 'r') as f_wet:
            x = torch.tensor(f_dry['Xtrain'][idx], dtype=torch.float32)
            y = torch.tensor(f_wet['Ytrain_0'][idx], dtype=torch.float32)
        return x, y


#-----------------------------------------------------------------------------#


import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Your PyTorch Lightning model and other training code...

# Set file paths
dry_train_path = 'dry_train.h5'
wet_train_path = 'wet_train.h5'
dry_val_test_path = 'dry_val_test.h5'
wet_val_test_path = 'wet_val_test.h5'

# Create datasets
train_dataset = H5AudioDataset(dry_train_path, wet_train_path)
val_dataset = H5AudioDataset(dry_val_test_path, wet_val_test_path)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Train the model
trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(model, train_loader, val_loader)