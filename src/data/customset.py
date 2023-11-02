import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torchaudio.functional as F
from pathlib import Path
import os


class CustomDataset(Dataset):
    """
    Custom dataset for the dry-wet audio pairs.
    Those can be a subset of another dataset or user-generated.

    The folder structure should be as follows:
    data_dir                # Path to the data directory referenced in the program arguments
    ├── customset           # Main folder for the custom dataset
        ├── input           # Folder for the dry/input/x audio files
        │   ├── 1-0.wav
        │   ├── 1-1.wav
        │   ├── ...
        ├── target          # Folder for the wet/target/y audio files
        │   ├── 1-0.wav
        │   ├── 1-1.wav
        │   ├── ...

    File names should be in the format "dry-<alphanumericID>.wav" and "wet-<alphanumericID>.wav".
    """

    def __init__(
        self,
        data_dir,
        transforms=None,
        sample_length=48000 * 4,
    ):
        self.data_dir = Path(data_dir) / "customset"
        self.input_dir = self.data_dir / "input"
        self.target_dir = self.data_dir / "target"

        self.transforms = transforms
        self.dry_files = []
        self.wet_files = []
        self.sample_length = sample_length

        # Divide files into dry and wet categories
        for file in self.input_dir.glob("*.wav"):
            self.dry_files.append(file)

        for file in self.target_dir.glob("*.wav"):
            self.wet_files.append(file)

        # Sort the file lists based on alphanumeric IDs
        self.dry_files.sort(key=lambda x: x.stem)
        self.wet_files.sort(key=lambda x: x.stem)

        assert len(self.dry_files) == len(
            self.wet_files
        ), "Dry and wet files must be paired with the same length."

    def __len__(self):
        return min(len(self.dry_files), len(self.wet_files))

    def load(self, audio_file):
        audio, sample_rate = torchaudio.load(audio_file, normalize=True)

        # Truncate audio to sample_length
        if audio.size(1) > self.sample_length:
            audio = audio[:, : self.sample_length]

        return audio

    def __getitem__(self, idx):
        dry_path = self.dry_files[idx]
        wet_path = self.wet_files[idx]

        # Load audio files
        dry_tensor = self.load(dry_path)
        wet_tensor = self.load(wet_path)

        if self.transforms:
            for transform in self.transforms:
                dry_tensor = transform(dry_tensor)
                wet_tensor = transform(wet_tensor)

        return dry_tensor, wet_tensor


# Transforms to apply to the audio files during loading


def contrast(tensor):
    return F.contrast(tensor, 50)


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


def load_customset(
    data_dir,
    batch_size,
    train_ratio=0.6,
    valid_ratio=0.2,
    test_ratio=0.2,
    num_workers=4,
):
    """Load and split the dataset"""
    dataset = CustomDataset(data_dir=data_dir, transforms=TRANSFORMS)

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
    train_data, valid_data, test_data = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(
        train_data,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate,
        pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        collate_fn=custom_collate,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        collate_fn=custom_collate,
        pin_memory=False,
    )

    return train_loader, valid_loader, test_loader
