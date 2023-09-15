import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.functional as F
from pathlib import Path
import glob
import os

class EgfxDataset(Dataset):
    """Egfx dataset
    Args:
        data_dir (str): Path to the data directory
        length (int): Length of the audio samples
        random_seed (int): Random seed for reproducibility
        transforms (list): List of transforms to apply to the audio samples
    
    Returns:
        torch.utils.data.Dataset: Dataset object containing tuples of dry and wet audio samples
    """
    def __init__(self,
                 data_dir,
                 sample_length=160000,
                 random_seed=42,
                 transforms=None
                ):
        self.data_dir = Path(data_dir) / 'egfxset'
        self.dry_dir = self.data_dir / 'Clean'
        self.wet_dir = self.data_dir / 'Spring Reverb'
        self.positions = ['Middle']
        
        torch.random.manual_seed(random_seed)
        
        self.dry_files = []
        self.wet_files = []

        for position in self.positions:
            dry_path = self.dry_dir / position
            wet_path = self.wet_dir / position
            dry_files_position = sorted(glob.glob(os.path.join(dry_path, '*.wav')))
            wet_files_position = sorted(glob.glob(os.path.join(wet_path, '*.wav')))

            if dry_files_position and wet_files_position:
                self.dry_files.extend(dry_files_position)
                self.wet_files.extend(wet_files_position)
            else:
                print(f"No files found in {position}")
        
        # Ensure that dry and wet files have the same length
        assert len(self.dry_files) == len(self.wet_files), "Dry and wet files must be paired with the same length."
        
        self.sample_length = sample_length
        self.transforms = transforms

    def __len__(self):
        return len(self.dry_files)

    def load(self, audio_file):
        audio, sample_rate = torchaudio.load(audio_file, normalize=True)
        
        # Truncate audio to sample_length
        if audio.size(1) > self.sample_length:
            audio = audio[:, :self.sample_length]
        return audio    
    
    def __getitem__(self, index):
        dry_file = self.dry_files[index]
        wet_file = self.wet_files[index]

        # print(Path(dry_file), Path(wet_file))
        # Load audio files
        dry_tensor = self.load(dry_file)
        wet_tensor = self.load(wet_file)

        # Transforms are provided as a list
        if self.transforms:
            for transform in self.transforms:
                dry_tensor = transform(dry_tensor)
                wet_tensor = transform(wet_tensor)
        
        return dry_tensor, wet_tensor


def contrast (tensor):
    return F.contrast(tensor, 75)

def correct_dc_offset(tensor):
    return tensor - torch.mean(tensor)

def peak_normalize(tensor):
    tensor /= torch.max(torch.abs(tensor))
    return tensor

def custom_collate(batch):
    # Separate the dry and wet samples
    dry, wet = zip(*batch)
    
    # Stack along the time dimension
    dry_stacked = torch.cat(dry, dim=-1)
    wet_stacked = torch.cat(wet, dim=-1)

    return dry_stacked.unsqueeze(0), wet_stacked.unsqueeze(0)

TRANSFORMS = [contrast, correct_dc_offset, peak_normalize]

def load_egfxset(data_dir, batch_size, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2, num_workers=4):
    """Load and split the dataset"""
    dataset = EgfxDataset(data_dir=data_dir, transforms=TRANSFORMS)

    # Calculate the sizes of train, validation, and test sets
    total_size = len(dataset)
    print(f"Total size: {total_size}")

    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = int(test_ratio * total_size)
    
    # Distribute the difference
    diff = total_size - (train_size + valid_size + test_size)
    train_size += diff

    # Split the dataset into train, validation, and test sets
    train_data, valid_data, test_data = torch.utils.data.random_split(dataset, [train_size, valid_size, test_size])

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(train_data, batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, collate_fn=custom_collate, pin_memory=False)
    valid_loader = DataLoader(valid_data, batch_size, num_workers=num_workers,
                              shuffle=False, drop_last=True, collate_fn=custom_collate, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, collate_fn=custom_collate, pin_memory=False)

    return train_loader, valid_loader, test_loader
