# src/helpers.py

import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path

from src.dataset_egfxset import EgfxDataset, CustomDataset
from src.networks.tcn import TCN
from src.networks.wavenet import WaveNetFF
from src.networks.lstm import LSTM, LstmConvSkip
from src.networks.gcn import GCN
from configurations import parse_args

args = parse_args()

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

def load_data(datadir, batch_size, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Load and split the dataset"""
    dataset = EgfxDataset(root_dir=datadir)

    # Calculate the sizes of train, validation, and test sets
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Split the dataset into train, validation, and test sets
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders for train, validation, and test sets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, num_workers=0, shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, num_workers=0, shuffle=False, drop_last=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, num_workers=0, drop_last=True, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


def select_device(device):
    if device is None: 
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"Selected device: {device}")
    
    return device


def initialize_model(device, hparams, args):
    if hparams['model_type'] == "TCN":
        model = TCN(
            n_inputs = 1,
            n_outputs = 1,
            n_blocks = hparams['n_blocks'],
            kernel_size = hparams['kernel_size'],
            num_channels = hparams['num_channels'], 
            dilation = hparams['dilation'],
            cond_dim = hparams['cond_dim'],
        ).to(device)
    elif hparams['model_type'] == "WaveNetFF":
        model = WaveNetFF(
            num_channels = hparams['num_channels'],
            dilation_depth = hparams['dilation_depth'],
            num_layers = hparams['num_layers'],
            kernel_size = hparams['kernel_size'],
        ).to(device)
    elif hparams['model_type'] == 'LSTM':
        model = LSTM(
            input_size = hparams['input_size'],
            output_size = hparams['output_size'],
            hidden_size = hparams['hidden_size'],
            num_layers = hparams['num_layers'],
        ).to(device)
    elif hparams['model_type'] == 'LstmConvSkip':
        model = LstmConvSkip(
            input_size = hparams['input_size'],
            hidden_size = hparams['hidden_size'],
            num_layers = hparams['num_layers'],
            output_size = hparams['output_size'],
            use_skip = hparams['use_skip'],
            kernel_size = hparams['kernel_size'],
        ).to(device)
    elif hparams['model_type'] == 'GCN':
        model = GCN(
            num_blocks = hparams['num_blocks'],
            num_layers = hparams['num_layers'],
            num_channels = hparams['num_channels'],
            kernel_size = hparams['kernel_size'],
            dilation_depth = hparams['dilation_depth'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {hparams['model_type']}")
    print(f"Model initialized: {hparams['conf_name']}")
    model.to(device)
    
    # Conditionally compute the receptive field for certain model types
    if hparams['model_type'] in ["TCN", "WaveNetFF", "GCN"]:
        rf = model.compute_receptive_field()
        print(f"Receptive field: {rf} samples or {(rf / args.sample_rate)*1e3:0.1f} ms", end='\n\n')    
    else:
        rf = None

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params*1e-3:0.3f} k")

    return model, rf , params

def load_model_checkpoint(device, checkpoint, args):
    checkpoint = torch.load(checkpoint, map_location=device)
    model_name = checkpoint['name']
    hparams = checkpoint['hparams']
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
    last_epoch = checkpoint.get('state_epoch', 0)  # If not found, we assume we start from epoch 0.
    model, _, _ = initialize_model(device, hparams, args)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model initialized: {model_name}")

    if hparams['model_type'] in ["TCN", "SimpleWaveNet"]:
        rf = model.compute_receptive_field()
        print(f"Receptive field: {rf} samples or {(rf / args.sample_rate)*1e3:0.1f} ms", end='\n\n')    
    else:
        rf = None
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params


def save_model_checkpoint(model, hparams, criterion, optimizer, scheduler, n_epochs, batch_size, lr, timestamp, avg_valid_loss, args):
    args = parse_args()
    model_type = hparams['model_type']
    model_name = hparams['conf_name']

    save_path = Path(args.modelsdir)/f"{model_type}"
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    save_to = save_path / f'{model_name}_{n_epochs}_{batch_size}_{timestamp}.pt'
    torch.save({
        'model_type': model_type,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'state_epoch': n_epochs,          
        'name': f'{model_name}_{n_epochs}_{batch_size}_{lr}_{timestamp}',
        'hparams': hparams,
        'criterion': str(criterion),
        'avg_valid_loss': avg_valid_loss,
    }, save_to)

def load_audio(input, sample_rate):
    print(f"Input type: {type(input)}")  # add this line to check the type of the input
    if isinstance(input, str):
        # Load audio file
        x_p, fs_x = torchaudio.load(input)
        x_p = x_p.float()
        print("sample rate: ", fs_x)
        input_name = Path(input).stem
    elif isinstance(input, np.ndarray):  # <-- change here
        # Resample numpy array if necessary
        # if input.shape[1] / sample_rate != len(input) / sample_rate:
        #     input = librosa.resample(input, input.shape[1], sample_rate)

        # Convert numpy array to tensor and ensure it's float32
        x_p = torch.from_numpy(input).float()
        # Add an extra dimension if necessary to simulate channel dimension
        if len(x_p.shape) == 1:
            x_p = x_p.unsqueeze(0)
        fs_x = sample_rate
        print("sample rate: ", fs_x)
        input_name = 'sweep'
    else:
        raise ValueError('input must be either a file path or a numpy array')

    # Ensure the audio is at the desired sample rate
    if fs_x != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=fs_x, new_freq=sample_rate)
        x_p = resampler(x_p)
        fs_x = sample_rate

        print("sample rate: ", fs_x)

    return x_p, fs_x, input_name

