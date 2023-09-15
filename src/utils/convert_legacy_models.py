from pyexpat import model
import sys
import os

project_path = '/gpfs/home/fpapaleo/neural-audio-spring-reverb'
os.environ['PYTHONPATH'] = f"{project_path}:{os.environ.get('PYTHONPATH', '')}"

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from datetime import datetime
import torch
from pathlib import Path
from argparse import ArgumentParser
from src.utils.checkpoints import initialize_model


def load_legacy_checkpoint(args):
    """Load a model checkpoint from a given path.

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

    checkpoint_data = torch.load(args.checkpoint, map_location=args.device)
    model_state_dict = checkpoint_data.get('model_state_dict')
    optimizer_state_dict = checkpoint_data.get('optimizer_state_dict', None)
    scheduler_state_dict = checkpoint_data.get('scheduler_state_dict', None)
    loaded_hparams = checkpoint_data['hparams']
    name = checkpoint_data['name']
    model_type = checkpoint_data['model_type']
    state_epoch = checkpoint_data['state_epoch']
    criterion = checkpoint_data['criterion']

    return checkpoint_data, model_state_dict, optimizer_state_dict, scheduler_state_dict, loaded_hparams, name, model_type, state_epoch, criterion    
    
def convert_legacy_checkpoint(args):

    # Define a list to store checkpoint paths
    checkpoint_paths = list(Path(args.models_dir).glob('*.pt'))
    print(f"Found {len(checkpoint_paths)} checkpoints in {args.models_dir}")

    checkpoint_data, model_state_dict, optimizer_state_dict, scheduler_state_dict, loaded_hparams, name, model_type, state_epoch, criterion  = load_legacy_checkpoint(args)
    
    print(f"{args.checkpoint}: {checkpoint_data.keys()}", end='\n\n')
    print(name)
    print(model_type)
    print(state_epoch)
    print(criterion)
    print(loaded_hparams)

    config = {
        'name': name,
        'model_type': model_type,

        'criterion1': 'mrstft',
        'pre_emphasis': None,
        'criterion2': None,

        'dataset': 'egfxset',
        'sample_rate': 48000,
        'bit_depth': 24,

        'optimizer': 'Adam',
        'lr': loaded_hparams['lr'],
        'lr_scheduler': 'ReduceLROnPlateau',
        'lr_patience': 25,

        'batch_size': 8,
        'num_workers': 8,

        'max_epochs': 1000,
        'early_stop_patience': 100,
        'current_epoch': 0,
        
        'min_valid_loss': None,
        'cond_dim': 0,

        # 'input_size': loaded_hparams['input_size'],
        # 'hidden_size': loaded_hparams['hidden_size'],
        # 'num_layers': loaded_hparams['num_layers'],
        # 'output_size': loaded_hparams['output_size'],
        # 'use_skip': True,
        # 'dropout': 0.5,
        'kernel_size': 9,

        'num_channels': loaded_hparams['n_channels'],
        'dilation_depth': loaded_hparams['dilation'],
        # 'num_blocks': loaded_hparams['n_blocks'],
        'num_repeat': loaded_hparams['num_repeat'],
    }

    print(config)

    model, rf, params = initialize_model(args.device, config)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sr_tag = str(int(config['sample_rate'] / 1000)) + 'k'
    label = f"{sr_tag}-{config['name']}-{config['criterion1']}.pt"

    save_path = Path(args.models_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    save_to = save_path / label

    # Save the model checkpoint in the new format:

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config['lr_patience'], verbose=True)

    torch.save({
        'label': label,
        'timestamp': timestamp,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config_state_dict': config,
    }, save_to)

    print("Done!")