import os

# Folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, 'dataset')
MODELS = os.path.join(BASE_DIR, 'models')
RESULTS = os.path.join(BASE_DIR, 'results')

LOCAL = '/Users/francescopapaleo/datasets/spring'

# Training loop parameters
cond_dim = 0
kernel_size = 9
n_blocks = 5
dilation_growth = 10
n_channels = 32
n_iters = 2500
length = 88800
lr = 0.001

# Define sample rate
sample_rate = 16000

# Define input and output channels
input_channels = 1
output_channels = 1