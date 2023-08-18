from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Train a TCN model on the plate-spring dataset')

    # Paths
    parser.add_argument('--datadir', type=str, default='../datasets/egfxset/', help='Path (rel) to the dataset ')
    parser.add_argument('--audiodir', type=str, default='audio/processed/', help='Path (rel) to the audio files')
    parser.add_argument('--logdir', type=str, default='results/', help='Path (rel) to  the log directory')
    parser.add_argument('--plotsdir', type=str, default='results/plots', help='Path (rel) to the plot directory')
    parser.add_argument('--checkpoint', type=str, default='results/checkpoints/', help='Path (rel) to checkpoint to load')
    parser.add_argument('--input', type=str, default=None, help='Path (rel) to audio file to process')

    # Model
    parser.add_argument('--config', type=str, default='TCN-BL-MRSTFT', help='The configuration to use')
    parser.add_argument('--device', type=str, default=None, help='set device to run the model on')
    parser.add_argument('--sample_rate', type=int, default=48000, help='sample rate of the audio')    
    parser.add_argument('--n_epochs', type=int, default=1000, help='the total number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    
    parser.add_argument('--c0', type=float, default=0.0, help='c0 parameter for the model')
    parser.add_argument('--c1', type=float, default=0.0, help='c1 parameter for the model')
    parser.add_argument('--mix', type=float, default=100.0, help='mix parameter for the model')
    
    # Measurements
    parser.add_argument("--duration", type=float, default=5.0, help="duration in seconds")
    parser.add_argument("--mode", type=str, choices=['ir', 'tf'], default='ir', help="Mode to run: 'ir' for impulse response or 'tf' for transfer function")
    
    return parser.parse_args()

configs = [{
    'conf_name': 'TCN-BL-MRSTFT',
    'model_type': 'TCN',
    'n_channels': 32,
    'dilation': 10,
    'n_blocks': 5,
    'kernel_size': 9,
    'cond_dim': 2,
    },
    {
    'conf_name': 'TCN-3',
    'model_type': 'TCN',
    'n_channels': 32,
    'dilation': 10,
    'n_blocks': 5,
    'kernel_size': 9,
    'cond_dim': 2,
    },
    {
    'conf_name': 'WN-MRSTFT',
    'model_type': 'WaveNet',
    'n_channels': 8,
    'dilation': 10,
    'num_repeat': 2,
    'kernel_size': 9,
    },
    {
    'conf_name': 'LSTM-32',
    'model_type': 'LSTM',
    'input_size': 1,
    'output_size': 1,
    'hidden_size': 32,
    'num_layers': 1,
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
}
"""