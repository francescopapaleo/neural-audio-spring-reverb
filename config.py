import os
import pathlib

# Folders
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(ROOT_DIR, 'dataset')
SUBSET = os.path.join(ROOT_DIR, 'dataset_subset')

AUDIO = os.path.join(ROOT_DIR, 'audio')
MODELS = os.path.join(ROOT_DIR, 'models')
RESULTS = os.path.join(ROOT_DIR, 'results')

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

# Model file name
model_trained = "model_TCN_02.pt"

# Evaluation parameters
model_to_evaluate = "model_TCN_01.pt"

# Number of parts to split the test data into
n_parts = 10

# Prediction
model_for_prediction = "model_TCN_01.pt"
