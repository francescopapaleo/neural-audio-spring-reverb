# helpers.py

import torch
import torchaudio
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from data.dataset import SpringDataset
from models.TCN import TCN
from models.WaveNet import WaveNet


def parse_args():
    parser = ArgumentParser(description='Train a TCN model on the plate-spring dataset')
    parser.add_argument('--datadir', type=str, default='../../datasets/plate-spring/spring/', help='Path (rel) to dataset ')
    parser.add_argument('--audiodir', type=str, default='../audio/processed/', help='Path (rel) to audio files')
    parser.add_argument('--logdir', type=str, default='../results/runs', help='name of the log directory')
    parser.add_argument('--checkpoint_path', type=str, default='../results/checkpoint', help='Path (rel) to checkpoint to load')
    parser.add_argument('--input', type=str, default=None, help='Path (rel) to audio file to process')

    parser.add_argument('--device', type=str, default=None, help='set device to run the model on')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate of the audio')    
    parser.add_argument('--n_epochs', type=int, default=2, help='the total number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--crop', type=int, default=3200, help='crop size')
    
    parser.add_argument('--max_length', type=float, default=None, help='maximum length of the output audio')
    parser.add_argument('--stereo', action='store_true', help='flag to indicate if the audio is stereo or mono')
    parser.add_argument('--tail', action='store_true', help='flag to indicate if tail padding is required')
    parser.add_argument('--width', type=float, default=50, help='width parameter for the model')
    parser.add_argument('--c0', type=float, default=0, help='c0 parameter for the model')
    parser.add_argument('--c1', type=float, default=0, help='c1 parameter for the model')
    parser.add_argument('--gain_dB', type=float, default=0, help='gain in dB for the model')
    parser.add_argument('--mix', type=float, default=50, help='mix parameter for the model')
    args = parser.parse_args()
    
    return parser.parse_args()


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


def load_data(datadir, batch_size):
    """Load and split the dataset"""
    trainset = SpringDataset(root_dir=datadir, split='train')
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    train, valid = torch.utils.data.random_split(trainset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=0, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=0, shuffle=False, drop_last=True)

    testset = SpringDataset(root_dir=datadir, split="test")
    test_loader = torch.utils.data.DataLoader(testset, batch_size, num_workers=0, drop_last=True)

    return train_loader, valid_loader, test_loader


def initialize_model(device, model_type, hparams=None, criterion=None, checkpoint_path=None):
    """Initialize a specific model"""

    if device is None: 
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # use the first cuda device
        else:
            device = torch.device("cpu")  # default to cpu if no cuda device is available
    else:
        device = torch.device(device)  # use the user-specified device

    # Load the model from a checkpoint if provided
    if checkpoint_path is not None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model = checkpoint['model_class'](
                n_inputs = checkpoint['hparams']['n_inputs'], 
                n_outputs = checkpoint['hparams']['n_outputs'], 
                n_blocks = checkpoint['hparams']['n_blocks'],
                kernel_size = checkpoint['hparams']['kernel_size'],
                n_channels = checkpoint['hparams']['n_channels'], 
                dilation_growth = checkpoint['hparams']['dilation_growth'],
                cond_dim = checkpoint['hparams']['cond_dim'],
            ).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise RuntimeError(f"Failed to load model state from checkpoint: {e}")
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    # Otherwise, initialize a new model as usual
    else:
        if model_type == "TCN":
            model = TCN(
                n_inputs = hparams['n_inputs'], 
                n_outputs = hparams['n_outputs'], 
                n_blocks = hparams['n_blocks'],
                kernel_size = hparams['kernel_size'],
                n_channels = hparams['n_channels'], 
                dilation_growth = hparams['dilation_growth'],
                cond_dim = hparams['cond_dim'],
            ).to(device)
        elif model_type == "WaveNet":
            model = WaveNet(
                n_inputs = hparams['n_inputs'], 
                n_outputs = hparams['n_outputs'], 
                n_blocks = hparams['n_blocks'],
                kernel_size = hparams['kernel_size'],
                n_channels = hparams['n_channels'], 
                dilation_growth = hparams['dilation_growth'],
                cond_dim = hparams['cond_dim'],
            ).to(device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    model.hparams = hparams  # store hyperparameters in the model
    model.criterion = str(criterion)  # store the name of the criterion in the model
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model, device, params


def save_model_checkpoint(model, hparams, criterion, optimizer, scheduler, n_epochs, batch_size, lr, timestamp):
    """Save the model if validation loss decreases"""
    save_to = f'../results/checkpoints/tcn_{n_epochs}_{batch_size}_{lr}_{timestamp}.pt'
    torch.save({
        'model_class': type(model),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'state_epoch': n_epochs,          
        'name': f'TCN{n_epochs}_{batch_size}_{lr}_{timestamp}',
        'hparams': hparams,
        'criterion': str(criterion)      # Add criterion as a string here
    }, save_to)

    print(f"Checkpoint saved: {save_to}")


