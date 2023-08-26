from pathlib import Path
import torch
import torchaudio
import glob
import os
import random

class EgfxDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 target_length=48000 * 5,
                 random_start=False,
                 random_seed=42
                ):
        self.root_dir = Path(root_dir) / 'egfxset'
        self.target_length = target_length
        self.random_start = random_start
        self.dry_dir = os.path.join(self.root_dir, 'Clean')
        self.wet_dir = os.path.join(self.root_dir, 'Spring Reverb')
        self.positions = ['Bridge', 'Bridge-Middle', 'Middle', 'Middle-Neck', 'Neck']
        
        random.seed(random_seed)  # set the seed for reproducibility

        self.dry_files = []
        self.wet_files = []

        for position in self.positions:
            dry_path = os.path.join(self.dry_dir, position)
            wet_path = os.path.join(self.wet_dir, position)
            dry_files_position = sorted(glob.glob(os.path.join(dry_path, '*.wav')))
            wet_files_position = sorted(glob.glob(os.path.join(wet_path, '*.wav')))

            if dry_files_position and wet_files_position:
                self.dry_files.extend(dry_files_position)
                self.wet_files.extend(wet_files_position)
            else:
                print(f"No files found in {position}")
        
        self.transform = transform

    def load(self, audio_file):
        audio, sample_rate = torchaudio.load(audio_file)
        return audio

    def __getitem__(self, index):
        dry_file = self.dry_files[index]
        wet_file = self.wet_files[index]

        # Load audio files
        dry_tensor = self.load(dry_file)
        wet_tensor = self.load(wet_file)

        # Truncate or pad the audio files to the target length
        if dry_tensor.size(1) > self.target_length:
            dry_tensor = dry_tensor[:, :self.target_length]
        if dry_tensor.size(1) < self.target_length:
            dry_tensor = torch.nn.functional.pad(dry_tensor, (0, self.target_length - dry_tensor.size(1)))

        # If random_start is set to True, randomly select a segment from the tensor
        if self.random_start:
            max_start_idx = self.target_length - dry_tensor.size(1)

            # Ensure the start index is non-negative
            max_start_idx = max(0, max_start_idx)
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + self.target_length

            dry_tensor = dry_tensor[:, start_idx:end_idx]
            wet_tensor = wet_tensor[:, start_idx:end_idx]

        # Return the audio data
        return dry_tensor, wet_tensor

    def __len__(self):
        return len(self.dry_files)


def peak_normalize(tensor):
    
    tensor /= torch.max(torch.abs(tensor))
    #  max_values = torch.max(torch.abs(tensor), dim=1, keepdim=True).values
    # normalized_tensor = tensor / max_values
    # return normalized_tensor
    # torch.nn.functional.normalize(tensor, p=2, dim=1)
    return tensor


def collate_fn(batch):
    # Separate the dry and wet samples
    dry_samples = [dry for dry, _ in batch]
    wet_samples = [wet for _, wet in batch]
    
    # Stack along the time dimension (dim=2 for 3D tensors)
    dry_stacked = torch.cat(dry_samples, dim=1)
    wet_stacked = torch.cat(wet_samples, dim=1)

    # Add an extra batch dimension 
    dry_stacked = dry_stacked.unsqueeze(0)
    wet_stacked = wet_stacked.unsqueeze(0)

    return dry_stacked, wet_stacked


def load_egfxset(datadir, batch_size, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25):
    """Load and split the dataset"""
    dataset = EgfxDataset(root_dir=datadir, transform=peak_normalize)

    # Calculate the sizes of train, validation, and test sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for train, validation, and test sets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, num_workers=0, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, num_workers=0, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, num_workers=0, drop_last=True)

    return train_loader, val_loader, test_loader