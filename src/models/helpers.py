import torch
from pathlib import Path
from src.models.tcn import TCN
from src.models.wavenet import PedalNetWaveNet
from src.models.lstm import LSTM, LstmConvSkip
from src.models.gcn import GCN
from src.models.bkp_wavenet import WaveNet
from src.default_args import parse_args

args = parse_args()

def select_device(device):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device

def initialize_model(device, hparams, conf_settings):
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
        print(f"Receptive field: {rf} samples or {(rf / conf_settings['sample_rate'])*1e3:0.1f} ms")    
    else:
        rf = None

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params*1e-3:0.3f} k")

    return model, rf, params

def load_model_checkpoint(device, checkpoint_path, args):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint['name']
    hparams = checkpoint['hparams']
    model_state_dict = checkpoint.get('model_state_dict')
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    scheduler_state_dict = checkpoint.get('scheduler_state_dict', None)
    conf_settings = checkpoint.get('conf_settings', None)
    if conf_settings is not None:
        conf_settings = checkpoint.get('conf_settings')
        state_epoch = conf_settings.get('state_epoch', 0)  # If not found, start from epoch 0
    else:
        conf_settings = {
            'sample_rate': args.sample_rate,
            'bit_rate': args.bit_rate,
            'max_epochs': args.max_epochs,
            'state_epoch': 0,
            'avg_valid_loss': None,
        }

    model, rf, params = initialize_model(device, hparams, conf_settings)
    model.load_state_dict(model_state_dict)
    
    return model, model_name, hparams, conf_settings, optimizer_state_dict, scheduler_state_dict, rf, params

def save_model_checkpoint(model, hparams, conf_settings, optimizer, scheduler, epoch, timestamp, avg_valid_loss):
    conf_name = hparams['conf_name']
    conf_settings.update({
            'max_epochs': args.max_epochs,
            'state_epoch': epoch,
            'avg_valid_loss': avg_valid_loss,
        })

    save_path = Path(args.models_dir)
    save_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
    sr_tag = args.sample_rate / 1000
    save_to = save_path / f'{conf_name}-{sr_tag}k.pt'
    
    torch.save({
        'name': f'{conf_name}_{timestamp}',
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'hparams': hparams,
        'conf_settings': conf_settings,
    }, save_to)
