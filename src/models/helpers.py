import torch
import copy
from pathlib import Path
from src.models.tcn import TCN
from src.models.wavenet import PedalNetWaveNet
from src.models.lstm import LSTM, LstmConvSkip
from src.models.gcn import GCN
from src.models.bkp_wavenet import WaveNet

def select_device(device):
    """
    Select the device to be used for training and inference.
    
    Parameters:
        device: str
            The device to be used. Can be either 'cuda' or 'cpu'.
    Returns:
        device: torch.device
            The device to be used for training and inference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device



def initialize_model(device, hparams):
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

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint.get('model_state_dict')
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
    hparams = checkpoint['hparams']

    model, rf, params = initialize_model(device, hparams)
    model.load_state_dict(model_state_dict)
    
    return model, optimizer_state_dict, scheduler_state_dict, hparams, rf, params



def save_model_checkpoint(model, hparams, optimizer, scheduler, current_epoch, timestamp, avg_valid_loss, args):
    """
    Save a model checkpoint to a specified path.

    Parameters:
        model : torch.nn.Module
            The model whose state needs to be saved.
        hparams : dict
            Hyperparameters dictionary associated with the model.
        optimizer : torch.optim.Optimizer
            The optimizer used during training of the model.
        scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler used during training.
        current_epoch : int
            The current epoch during training when the checkpoint is saved.
        timestamp : str
            Timestamp indicating when the checkpoint was created or last modified.
        avg_valid_loss : float
            The average validation loss at the time of saving.
        args : 
            Additional arguments or configurations. Must have attributes 'sample_rate' and 'models_dir'.

    Returns:
        None. The function saves the checkpoint to the designated path.
    """
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
    