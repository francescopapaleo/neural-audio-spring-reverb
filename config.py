from pathlib import Path

seed = 42

# Folders
ROOT_DIR = Path(__file__).resolve().parent

# DATASET_DIR = ROOT_DIR / 'dataset'
# DATA_DIR = '/Users/francescopapaleo/datasets/plate-spring/spring'
DATA_DIR = ROOT_DIR / 'data'

AUDIO_DIR = ROOT_DIR / 'audio'
MODELS_DIR = ROOT_DIR / 'models'
RESULTS = ROOT_DIR / 'results'
IMGS_DIR = ROOT_DIR / 'imgs'

# Model parameters
model_params = {
    "cond_dim": 0,
    "kernel_size": 9,
    "n_blocks": 5,
    "dilation_growth": 10,
    "n_channels": 32,
    "n_iters": 50,
    "length": 88800,
    "lr": 0.001,
    "batch_size": 1,
    "c": 0.0
}

# Set sample rate
SAMPLE_RATE = 16000
INPUT_CH = 1
OUTPUT_CH = 1

# Trained model filename
MODEL_FILE = MODELS_DIR / "model_TCN_01.pt"

# For inference
OUTPUT_FILE = AUDIO_DIR / "processed.wav"

# Processing parameters
processing_params = {
    "gain_dB": -0.1,
    "c0": 0.6,
    "c1": 0,
    "mix": 100,
    "width": 21,
    "max_length": 30,
    "stereo": False,
    "tail": True,
    "output_file": OUTPUT_FILE
}
