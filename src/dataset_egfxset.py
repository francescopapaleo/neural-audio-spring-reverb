# src/egfxset.py

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
                 random_start=True,
                 random_seed=42
                ):
        self.root_dir = root_dir
        self.random_start = random_start
        self.dry_dir = os.path.join(self.root_dir, 'Clean')
        self.wet_dir = os.path.join(self.root_dir, 'Spring Reverb')
        self.positions = ['Bridge', 'Bridge-Middle', 'Middle', 'Middle-Neck', 'Neck']
        
        self.target_length = target_length
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

    def load(self, audio_file):
        audio, sample_rate = torchaudio.load(audio_file)
        return audio

    def __getitem__(self, index):
        dry_file = self.dry_files[index]
        wet_file = self.wet_files[index]

        # Load and normalize the audio files
        dry_tensor = self.load(dry_file)
        wet_tensor = self.load(wet_file)

        # Get the lengths of the two tensors
        dry_length = dry_tensor.size(1)
        wet_length = wet_tensor.size(1)
        
        # If dry_tensor is longer than wet_tensor, truncate or pad wet_tensor
        if dry_length > wet_length:
            padding_size = dry_length - wet_length
            wet_tensor = torch.nn.functional.pad(wet_tensor, (0, padding_size))
        # If wet_tensor is longer than dry_tensor, truncate or pad dry_tensor
        elif wet_length > dry_length:
            padding_size = wet_length - dry_length
            dry_tensor = torch.nn.functional.pad(dry_tensor, (0, padding_size))

        if self.random_start and (dry_tensor.size(1) > self.target_length):
            max_start_idx = dry_tensor.size(1) - self.target_length
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + self.target_length
            
            dry_tensor = dry_tensor[:, start_idx:end_idx]
            wet_tensor = wet_tensor[:, start_idx:end_idx]

        # Return the audio data
        return dry_tensor, wet_tensor

    def __len__(self):
        return len(self.dry_files)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 target_length=48000 * 8,
                 random_start=False,
                 random_seed=42
                ):
        self.root_dir = root_dir
        self.random_start = random_start
        self.dry_dir = os.path.join(self.root_dir, 'dry')
        self.wet_dir = os.path.join(self.root_dir, 'wet')
        
        self.target_length = target_length
        random.seed(random_seed)  # set the seed for reproducibility

        self.dry_files = sorted(glob.glob(os.path.join(self.dry_dir, '*.wav')))
        self.wet_files = sorted(glob.glob(os.path.join(self.wet_dir, '*.wav')))

        assert len(self.dry_files) == len(self.wet_files), "The number of dry and wet files should be the same."

    def load_and_normalize(self, audio_file):
        audio, sample_rate = torchaudio.load(audio_file)
        # Normalize audio
        # normalized_audio = audio / torch.abs(audio).max()
        normalized_audio = audio
        # Convert to PyTorch tensor and add channel dimension
        return normalized_audio

    def __getitem__(self, index):
        dry_file = self.dry_files[index]
        wet_file = self.wet_files[index]

        # Load and normalize the audio files
        dry_tensor = self.load_and_normalize(dry_file)
        wet_tensor = self.load_and_normalize(wet_file)

        # Get the lengths of the two tensors
        dry_length = dry_tensor.size(1)
        wet_length = wet_tensor.size(1)
        
        # If dry_tensor is longer than wet_tensor, truncate or pad wet_tensor
        if dry_length > wet_length:
            padding_size = dry_length - wet_length
            wet_tensor = torch.nn.functional.pad(wet_tensor, (0, padding_size))
        # If wet_tensor is longer than dry_tensor, truncate or pad dry_tensor
        elif wet_length > dry_length:
            padding_size = wet_length - dry_length
            dry_tensor = torch.nn.functional.pad(dry_tensor, (0, padding_size))

        if self.random_start and (dry_tensor.size(1) > self.target_length):
            max_start_idx = dry_tensor.size(1) - self.target_length
            start_idx = random.randint(0, max_start_idx)
            end_idx = start_idx + self.target_length
            
            dry_tensor = dry_tensor[:, start_idx:end_idx]
            wet_tensor = wet_tensor[:, start_idx:end_idx]

        # Return the audio data
        return dry_tensor, wet_tensor

    def __len__(self):
        return len(self.dry_files)