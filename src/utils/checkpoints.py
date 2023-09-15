import torch
import yaml
from pathlib import Path
from datetime import datetime

from src.networks.tcn import TCN
from src.networks.wavenet import WaveNet
from src.networks.gcn import GCN
from src.networks.lstm import LSTM
from src.networks.gru import GRU
from src.networks.lstm_cs import LSTM_CS
from src.networks.gcn_film import GCN_FiLM
from src.networks.lstm_film import LSTM_FiLM

# DEVICE SELECTION HAS BEEN MOVED TO main.py
# def select_device(device):
#     """
#     Select the device to be used for training and inference.
    
#     Parameters:
#         device: str
#             The device to be used. Can be either 'cuda:0' or 'cpu'.
#     Returns:
#         device: torch.device
#             The device to be used for training and inference.
#     """
#     if device is not None:
#         device = torch.device(device)
#     elif torch.cuda.is_available():
#         device = torch.device("cuda:0")
#     else:
#         device = torch.device("cpu")
#     print(f"Using device: {device}")
#     return device


def parse_config(config_path):
    """
    Parse a YAML configuration file into a dictionary.
    """
    with open(config_path, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
        return dict(config_dict)


def initialize_model(device, config):
    """
    Initialize a model based on the model type specified in the hparams.

    Parameters:
        device: torch.device
            The device (e.g., 'cuda' or 'cpu') to which the model should be transferred.
        hparams: dict
            Hyperparameters dictionary containing settings and specifications for the model. 
        
    Returns:
        model: torch.nn.Module 
            nitialized model of the type specified in hparams.
        rf: int or None
            Receptive field of the model in terms of samples. 
            Only computed for specific model types ["TCN", "PedalNetWaveNet", "GCN"].
            Returns None for other model types.
        params: int
            Total number of trainable parameters in the model.
    """
    model_dict = {
        'TCN': TCN,
        'LSTM_FiLM': LSTM_FiLM,
        'GCN_FiLM': GCN_FiLM,
        'WaveNet': WaveNet,
        'LSTM': LSTM,
        'GCN': GCN,
        'GRU': GRU,
        'LstmConvSkip': LSTM_CS,
    }

    model_params = {      
        'TCN': {'num_blocks', 'kernel_size', 'num_channels', 'dilation_depth', 'cond_dim'},
        'GCN_FiLM': {'num_blocks', 'num_layers', 'num_channels', 'kernel_size', 'dilation_depth', 'cond_dim'},
        'LSTM_FiLM': {'input_size', 'hidden_size', 'num_layers', 'output_size', 'use_skip', 'kernel_size', 'cond_dim'},
        'GCN': {'num_blocks', 'num_layers', 'num_channels', 'kernel_size', 'dilation_depth', 'cond_dim'},
        'WaveNet': {'num_channels', 'dilation_depth', 'num_repeat', 'kernel_size', 'cond_dim'},
        'GRU': {'input_size', 'hidden_size', 'num_layers', 'output_size', 'dropout_prob', 'use_skip', 'kernel_size', 'cond_dim'},
        'LstmConvSkip': {'input_size', 'hidden_size', 'num_layers', 'output_size', 'use_skip', 'kernel_size', 'cond_dim'},
        'LSTM': {'input_size', 'hidden_size', 'num_layers', 'output_size', 'use_skip', 'kernel_size', 'cond_dim'},
    }

    if config['model_type'] not in model_dict:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    # Filter hparams to only include the keys specific to the model type
    filtered_hparams = {k: v for k, v in config.items() if k in model_params[config['model_type']]}

    model = model_dict[config['model_type']](**filtered_hparams).to(device)
    print(f"Configuration name: {config['name']}")
    
    # Conditionally compute the receptive field for certain model types
    if config['model_type'] in ["TCN", "PedalNetWaveNet", "GCN"]:
        rf = model.compute_receptive_field()

        print(f"Receptive field: {rf} samples or {(rf / config['sample_rate'])*1e3:0.1f} ms")    
    else:
        rf = None

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params*1e-3:0.3f} k")

    return model, rf, params


def load_model_checkpoint(args):
    """
    Load a model checkpoint from a given path.

    Parameters:
        device : torch.device
            The device (e.g., 'cuda' or 'cpu') where the checkpoint will be loaded.
        checkpoint_path : str
            Path to the checkpoint file to be loaded.
        args : 
            Additional arguments or configurations (currently unused in the function but can be 
            utilized for future extensions).

    Returns:
        model : torch.nn.Module
            Initialized and state-loaded model from the checkpoint.
        optimizer_state_dict : dict or None
            State dictionary for the optimizer if present in the checkpoint; None otherwise.
        scheduler_state_dict : dict or None
            State dictionary for the learning rate scheduler if present in the checkpoint; None otherwise.
        hparams : dict
            Hyperparameters dictionary loaded from the checkpoint.
        rf : int or None
            Receptive field of the model in terms of samples, computed during model initialization. 
            Only computed for specific model types; None for others.
        params : int
            Total number of trainable parameters in the model.
    """

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_state_dict = checkpoint.get('model_state_dict')
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
    loaded_config = checkpoint['config_state_dict']

    model, rf, params = initialize_model(args.device, loaded_config)
    model.load_state_dict(model_state_dict)
    
    return model, optimizer_state_dict, scheduler_state_dict, loaded_config, rf, params



def save_model_checkpoint(model, config, optimizer, scheduler, current_epoch, label, min_valid_loss, args):
    """
    Save a model checkpoint to a specified path.

    Parameters:
        model : torch.nn.Module
            The model whose state needs to be saved.
        
        optimizer : torch.optim.Optimizer
            The optimizer used during training of the model.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler used during training.
        current_epoch : int
            The current epoch during training when the checkpoint is saved.
        timestamp : str
            Timestamp indicating when the checkpoint was created or last modified.
        
        args : 
            Additional arguments or configurations. Must have attributes 'sample_rate' and 'models_dir'.

    Returns:
        None. The function saves the checkpoint to the designated path.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    config.update({
        'current_epoch': current_epoch,
        'timestamp': timestamp,
        'min_valid_loss': min_valid_loss,
    })
    file_name = f"{label}.pt"
    save_path = Path(args.models_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_to = save_path / file_name

    torch.save({
        'label': label,
        'timestamp': timestamp,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config_state_dict': config,
    }, save_to)