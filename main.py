import json
from pathlib import Path

# Folders
MAIN_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(MAIN_DIR / 'data')
AUDIO_DIR = Path(MAIN_DIR / 'audio')

MODELS_DIR = Path(MAIN_DIR / 'models')
RESULTS = Path(MAIN_DIR / 'results')

OUTPUT_FILE = Path(RESULTS / "processed.wav")

# Set sample rate
fs = 16000

model_params = {
    "cond_dim": 0,
    "n_blocks": 5,
    "dilation_growth": 10,
    "kernel_size": 9,
    "n_channels": 32,
    "n_iters": 1000,
    "length": 88800,
    "lr": 0.001,
    "batch_size": 1,
    "c": 0.0,
    'in_ch': 1,
    'out_ch': 1,
    "gain_dB": -0.1,
    "c0": 0.6,
    "c1": 0,
    "mix": 100,
    "width": 21,
    "max_length": 30,
    "stereo": False,
    "tail": True
    }