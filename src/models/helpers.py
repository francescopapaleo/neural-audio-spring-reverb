import torch
import copy
from pathlib import Path
from src.models.tcn import TCN
from src.models.wavenet import PedalNetWaveNet
from src.models.lstm import LSTM, LstmConvSkip
from src.models.gcn import GCN
from src.models.bkp_wavenet import WaveNet

def select_device(device):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def initialize_model(device, hparams):
    model_dict = {
        'TCN': TCN,
        'PedalNetWaveNet': PedalNetWaveNet,
        'LSTM': LSTM,
        'LstmConvSkip': LstmConvSkip,
        'GCN': GCN,
        'WaveNet': WaveNet
    }

    model_params = {
        'TCN': {'n_blocks', 'kernel_size', 'num_channels', 'dilation', 'cond_dim'},
        'PedalNetWaveNet': {'num_channels', 'dilation_depth', 'num_repeat', 'kernel_size'},
        'LSTM': {'input_size', 'output_size', 'hidden_size', 'num_layers'},
        'LstmConvSkip': {'input_size', 'hidden_size', 'num_layers', 'output_size', 'use_skip', 'kernel_size'},
        'GCN': {'num_blocks', 'num_layers', 'num_channels', 'kernel_size', 'dilation_depth'},
        'WaveNet': {'n_channels', 'dilation', 'num_repeat', 'kernel_size'}
    }

    if hparams['model_type'] not in model_dict:
        raise ValueError(f"Unknown model type: {hparams['model_type']}")

    # Filter hparams to only include the keys specific to the model type
    filtered_hparams = {k: v for k, v in hparams.items() if k in model_params[hparams['model_type']]}

    model = model_dict[hparams['model_type']](**filtered_hparams).to(device)
    print(f"Configuration name: {hparams['conf_name']}")
    
    # Conditionally compute the receptive field for certain model types
    if hparams['model_type'] in ["TCN", "PedalNetWaveNet", "GCN"]:
        rf = model.compute_receptive_field()
        print(f"Receptive field: {rf} samples or {(rf / hparams['sample_rate'])*1e3:0.1f} ms")    
    else:
        rf = None

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params*1e-3:0.3f} k")

    return model, rf, params

def load_model_checkpoint(device, checkpoint_path, args):
    """Load a model checkpoint"""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint.get('model_state_dict')
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
    hparams = checkpoint['hparams']

    model, rf, params = initialize_model(device, hparams)
    model.load_state_dict(model_state_dict)
    
    return model, optimizer_state_dict, scheduler_state_dict, hparams, rf, params


def save_model_checkpoint(model, hparams, optimizer, scheduler, current_epoch, timestamp, avg_valid_loss, args):
    """Save a model checkpoint"""

    updated_hparams = copy.deepcopy(hparams)
    updated_hparams.update({
            'state_epoch': current_epoch,
            'avg_valid_loss': avg_valid_loss,
    })
    
    sr_tag = str(int(args.sample_rate / 1000)) + 'kHz'
    save_as = hparams['conf_name'] + '_' + hparams['criterion'] + '-' + sr_tag + '.pt'
    save_path = Path(args.models_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_to = save_path / save_as

    torch.save({
        'timestamp': timestamp,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'hparams': updated_hparams,
    }, save_to)

