from pathlib import Path

seed = 42

# Folders
ROOT_DIR = Path(__file__).resolve().parent

# DATASET_DIR = ROOT_DIR / 'dataset'
# DATA_DIR = '/Users/francescopapaleo/datasets/plate-spring/spring'
DATA_DIR = ROOT_DIR / 'data'
AUDIO_DIR = ROOT_DIR / 'audio'
RESULTS = ROOT_DIR / 'results'

# Model parameters
model_params = {
    "cond_dim": 0,
    "kernel_size": 9,
    "n_blocks": 5,
    "dilation_growth": 10,
    "n_channels": 32,
    "n_iters": 1000,
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
MODEL_PATH = '/homedtic/fpapaleo/smc-spring-reverb/models/tcn_1000.pt'
# MODELS_DIR = Path(ROOT_DIR / 'models')
# MODEL_FILE = "tcn_1000.pt"

# For inference
OUTPUT_FILE = RESULTS / "processed.wav"

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
