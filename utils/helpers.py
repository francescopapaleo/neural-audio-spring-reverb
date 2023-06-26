# helpers.py

import torch
import torchaudio
import numpy as np
from pathlib import Path

from data.dataset import SpringDataset
from models.TCN import TCN
from models.WaveNet import WaveNet
from config import parse_args

def load_audio(input, sample_rate):
    print(f"Input type: {type(input)}")  # add this line to check the type of the input
    if isinstance(input, str):
        # Load audio file
        x_p, fs_x = torchaudio.load(input)
        x_p = x_p.float()
        input_name = Path(input).stem
    elif isinstance(input, np.ndarray):  # <-- change here
        # Convert numpy array to tensor and ensure it's float32
        x_p = torch.from_numpy(input).float()
        # Add an extra dimension if necessary to simulate channel dimension
        if len(x_p.shape) == 1:
            x_p = x_p.unsqueeze(0)
        fs_x = sample_rate
        input_name = 'sweep'
    else:
        raise ValueError('input must be either a file path or a numpy array')
    
    return x_p, fs_x, input_name

def peak_normalize(tensor):
    return tensor / tensor.abs().max()

def load_data(datadir, batch_size):
    """Load and split the dataset"""
    trainset = SpringDataset(root_dir=datadir, split='train', transform=peak_normalize)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train, valid = torch.utils.data.random_split(trainset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=0, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=0, shuffle=False, drop_last=True)

    testset = SpringDataset(root_dir=datadir, split="test", transform=peak_normalize)
    test_loader = torch.utils.data.DataLoader(testset, batch_size, num_workers=0, drop_last=True)

    return train_loader, valid_loader, test_loader


def select_device(device):
    if device is None: 
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"Selected device: {device}")
    
    return device


def initialize_model(device, hparams):
    args = parse_args()
    sample_rate = args.sample_rate
    if hparams['model_type'] == "TCN":
        model = TCN(
            n_inputs = hparams['n_inputs'], 
            n_outputs = hparams['n_outputs'], 
            n_blocks = hparams['n_blocks'],
            kernel_size = hparams['kernel_size'],
            n_channels = hparams['n_channels'], 
            dilation_growth = hparams['dilation_growth'],
            cond_dim = hparams['cond_dim'],
        ).to(device)
    elif hparams['model_type'] == "WaveNet":
        model = WaveNet(
            num_channels = hparams['num_channels'],
            dilation_depth=hparams['dilation_depth'],
            num_repeat=hparams['num_repeat'],
            kernel_size = hparams['kernel_size'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {hparams['model_type']}")
    print(f"Model initialized: {hparams['model_type']}")
    model.to(device)
    
    rf = model.compute_receptive_field()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {params*1e-3:0.3f} k")
    print(f"Receptive field: {rf} samples or {(rf / sample_rate)*1e3:0.1f} ms", end='\n\n')    

    return model, rf , params


def load_model_checkpoint(device, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_name = checkpoint['name']

        hparams = checkpoint['hparams']
        model, _, _ = initialize_model(device, hparams)

        model.load_state_dict(checkpoint['model_state_dict'])
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model state from checkpoint: {e}")
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    
    return model, model_name, hparams


def save_model_checkpoint(model, hparams, criterion, optimizer, scheduler, n_epochs, batch_size, lr, timestamp):
    args = parse_args()
    model_type = hparams['model_type']
    save_to = Path(args.checkpoint_path) / f'{model_type}_{n_epochs}_{batch_size}_{lr}_{timestamp}.pt'
    torch.save({
        'model_type': model_type,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'state_epoch': n_epochs,          
        'name': f'TCN{n_epochs}_{batch_size}_{lr}_{timestamp}',
        'hparams': hparams,
        'criterion': str(criterion)
    }, save_to)

    print(f"Checkpoint saved: {save_to}")

