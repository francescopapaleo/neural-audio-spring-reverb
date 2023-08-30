from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='')

    # Paths
    parser.add_argument('--datadir', type=str, default='../datasets', help='Path (rel) to the dataset ')
    parser.add_argument('--audiodir', type=str, default='audio', help='Path (rel) to the audio files')
    parser.add_argument('--logdir', type=str, default='results/', help='Path (rel) to  the log directory')
    parser.add_argument('--plotsdir', type=str, default='results/plots', help='Path (rel) to the plot directory')
    parser.add_argument('--modelsdir', type=str, default='results/', help='Path (rel) to models checkpoints directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path (rel) to checkpoint file')
    parser.add_argument('--input', type=str, default=None, help='Path (rel) to audio file to process')
    parser.add_argument('--dataset', type=str, default='egfxset', help='Dataset to use')
    
    # Audio
    parser.add_argument('--sample_rate', type=int, default=48000, help='sample rate of the audio') 
    parser.add_argument('--bit_rate', type=int, default=24, help='bits per second')  

    # Model
    parser.add_argument('--config', type=str, default=None, help='The configuration to use')
    parser.add_argument('--device', type=str, default='cuda:0', help='set device to run the model on')
 
    # Training
    parser.add_argument('--n_epochs', type=int, default=100, help='the total number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    
    parser.add_argument('--mix', type=float, default=100.0, help='mix parameter for the model')
    
    # Measurements
    parser.add_argument("--duration", type=float, default=5.0, help="duration in seconds")
    
    return parser.parse_args()

configs = [
    {
    'conf_name': 'tcn-5500',
    'model_type': 'TCN',
    'n_blocks': 5,
    'kernel_size': 9,
    'num_channels': 32,
    'dilation': 10,
    'cond_dim': 0,
    'criterion': 'mae+mrstft',
    },
    {
    'conf_name': 'tcn-4000',
    'model_type': 'TCN',
    'n_blocks': 5,
    'kernel_size': 19,
    'num_channels': 32,
    'dilation': 10,
    'cond_dim': 0,
    'criterion': 'esr+dc',
    },
    {
    'conf_name': 'gcn-250-v30',
    'model_type': 'GCN',
    'num_blocks': 1,
    'num_layers': 4,
    'num_channels': 16,
    'kernel_size': 41,
    'dilation_depth': 6,
    'criterion': 'mae+mrstft',
    },
    {
    'conf_name': 'gcn-2500',
    'model_type': 'GCN',
    'num_blocks': 1,
    'num_layers': 10,
    'num_channels': 16,
    'kernel_size': 5,
    'dilation_depth': 3,
    'criterion': 'mae+mrstft',
    },
    {
    'conf_name': 'wavenet-10',
    'model_type': 'PedalNetWaveNet',
    'num_channels': 16,
    'dilation_depth': 10,
    'num_repeat': 1,
    'kernel_size': 3,
    'criterion': 'mae+mrstft',
    },
    {
    'conf_name': 'wavenet-18',
    'model_type': 'PedalNetWaveNet',
    'num_channels': 16,
    'dilation_depth': 9,
    'num_repeat': 2,
    'kernel_size': 3,
    'criterion': 'mae+mrstft',
    },
    {
    'conf_name': 'wavenet-24',
    'model_type': 'PedalNetWaveNet',
    'num_channels': 16,
    'dilation_depth': 8,
    'num_repeat': 3,
    'kernel_size': 3,
    'criterion': 'mae+mrstft',
    },
    {
    'conf_name': 'wavenet-1k5-v28',
    'model_type': 'PedalNetWaveNet',
    'num_channels': 8,
    'dilation_depth': 10,
    'num_repeat': 5,
    'kernel_size': 5,
    'criterion': 'mae+mrstft',
    },
    {
    'conf_name': 'LSTM-96',
    'model_type': 'LSTM',
    'input_size': 1,
    'hidden_size': 96,
    'num_layers': 1,
    'output_size': 1,
    'criterion': 'mae+mrstft'
    },
    {
    'conf_name': 'lstm-cs-32',
    'model_type': 'LstmConvSkip',
    'input_size': 1,
    'hidden_size': 32,
    'num_layers': 2,
    'output_size': 1,
    'use_skip': True,
    'kernel_size': 3,
    'criterion': 'mae+mrstft',
    },
    {
    'conf_name': 'lstm-cs-96-v28',
    'model_type': 'LstmConvSkip',
    'input_size': 1,
    'hidden_size': 96,
    'num_layers': 1,
    'output_size': 1,
    'use_skip': True,
    'kernel_size': 3,
    'criterion': 'mae+mrstft',
    },

    ]


"""TODO: Add more configurations with the following format:
{
name='tcn-base',
type='tcn',
in_size=1,          # Number of input channels (mono = 1, stereo 2)
out_size=1,         # Number of output channels (mono = 1, stereo 2)
cond_dim=0,         # Number of conditioning parameters
n_blocks=10,        # Number of total TCN blocks
n_layers=1,         
n_channels=32,
kernel_size=3,      # Width of the convolutional kernels 
dilation_growth=1,  # Compute the dilation factor at each block as dilation_growth ** (n % stack_size)
channel_growth=1,   # Compute the output channels at each black as in_ch * channel_growth 
channel_width=1,    # When channel_growth = 1 all blocks use convolutions with this many channels
stack_size=1,       # Number of blocks that constitute a single stack of blocks
grouped=False,      # Use grouped convolutions to reduce the total number of parameters
causal=False,       # Causal TCN configuration does not consider future input values
skip_connections=False,     # Skip connections from each block to the output
loss_fn='mrstft',           # Loss function to use
pre_filt="high_pass",       # Pre-filtering to apply to the input
batch_size=1,               # Batch size
n_epochs=1000,
n_workers=0,
lr=1e-3,
sample_rate=48000,
bit_rate=24,
}
"""