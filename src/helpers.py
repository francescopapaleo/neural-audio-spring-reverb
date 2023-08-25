import torch
from pathlib import Path

<<<<<<< HEAD
from src.dataset import SpringDataset
from src.networks.TCN import TCN
from src.networks.WaveNet import WaveNet
from src.networks.LSTM import LSTM, LSTMskip
from configurations import parse_args


import librosa

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



def peak_normalize(tensor):
    # max_values = torch.max(torch.abs(tensor), dim=1, keepdim=True).values
    # normalized_tensor = tensor / max_values
    # return normalized_tensor
    torch.nn.functional.normalize(tensor, p=2, dim=1)
    return tensor


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

=======
from src.networks.tcn import TCN
from src.networks.wavenet import PedalNetWaveNet
from src.networks.lstm import LSTM, LstmConvSkip
from src.networks.gcn import GCN
from configurations import parse_args

args = parse_args()
>>>>>>> 48kHz

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
