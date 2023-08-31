import torch
from pathlib import Path

from src.networks.tcn import TCN
from src.networks.wavenet import PedalNetWaveNet
from src.networks.lstm import LSTM, LstmConvSkip
from src.networks.gcn import GCN
from src.networks.bkp_wavenet import WaveNet
from configurations import parse_args

args = parse_args()

def select_device(device):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
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
    elif hparams['model_type'] == "PedalNetWaveNet":
        model = PedalNetWaveNet(
            num_channels = hparams['num_channels'],
            dilation_depth = hparams['dilation_depth'],
            num_repeat = hparams['num_repeat'],
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
    elif hparams['model_type'] == "WaveNet":
        model = WaveNet(
            n_channels = hparams['n_channels'],
            dilation=hparams['dilation'],
            num_repeat=hparams['num_repeat'],
            kernel_size = hparams['kernel_size'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {hparams['model_type']}")
    print(f"Model initialized: {hparams['conf_name']}")
    model.to(device)
    
    # Conditionally compute the receptive field for certain model types
    if hparams['model_type'] in ["TCN", "PedalNetWaveNet", "GCN"]:
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


def save_model_checkpoint(
    model, hparams, optimizer, scheduler, epoch, timestamp, avg_valid_loss, args
    ):
    args = parse_args()
    model_type = hparams['model_type']
    model_name = hparams['conf_name']

    hparams.update({
            'curr_epoch': epoch,
        })

    save_path = Path(args.modelsdir)
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    sr_tag = args.sample_rate / 1000

    save_to = save_path / f'{model_name}-{sr_tag}k.pt'
    torch.save({
        'model_type': model_type,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'state_epoch': epoch,          
        'name': f'{model_name}_{timestamp}',
        'hparams': hparams,
        'avg_valid_loss': avg_valid_loss,
    }, save_to)
