import os
from pathlib import Path

# Folders
ROOT_DIR = Path(__file__).resolve().parent

DATASET = ROOT_DIR / 'dataset'
SUBSET = ROOT_DIR / 'dataset_subset'

AUDIO = ROOT_DIR / 'audio'
MODELS = ROOT_DIR / 'models'
RESULTS = ROOT_DIR / 'results'

# Training loop parameters
cond_dim = 0
kernel_size = 9
n_blocks = 5
dilation_growth = 10
n_channels = 32
n_iters = 50
length = 88800
lr = 0.001

# Set sample rate
sample_rate = 16000

# Define number of input and output channels
input_channels = 1
output_channels = 1

# Trained model filename
model_trained = "model_TCN_00.pt"

# Evalute model
model_to_evaluate = "model_TCN_00.pt"

# Prediction
model_for_prediction = "reverb_full.pt"

# Number of parts to split the test data into
n_parts = 10

