# config.py

from argparse import ArgumentParser

from pathlib import Path

parser = ArgumentParser()

model_params = {
    "cond_dim": 0,
    "n_blocks": 5,
    "dilation_growth": 10,
    "kernel_size": 6,
    "n_channels": 32,
    "length": 88800,
    "lr": 0.001,
    "batch_size": 1,
    "c": 0.0,
    "gain_dB": -0.1,
    "c0": 0.6,
    "c1": 0,
    "mix": 100,
    "width": 21,
    "max_length": 30,
    "stereo": False,
    "tail": True
    }

# PROGRAM level args
parser.add_argument('--data_dir', type=str, default='./data/plate-spring/spring/', 
                    help='default dataset folder')

parser.add_argument('--models_dir', type=str, default='./models', 
                    help='folder to store state_dict after training and load for eval or inference')

parser.add_argument('--results_dir', type=str, default='./results', 
                    help='folder to store results/processed files')

parser.add_argument('--sr', type=int, default=16000, 
                    help='sampling rate frequency, default: 16KHz')


parser.add_argument('--target_dir', type=str, 
                    help='set target folder for a specific function')

parser.add_argument('--target_file', type=str, 
                    help='set target file for a specific function ' )


parser.add_argument('--save', type=str, default='model.pt', 
                    help='save weights and biases as')

parser.add_argument('--load', type=str, 
                    help='load weights and biases from')


parser.add_argument('--device', type=str, default='cpu', 
                    help='set the device for training and inference')

parser.add_argument('--input', type=str, 
                    help="input file to process")

parser.add_argument('--sample_idx', type=int, default=0, 
                    help='The index of the sample from a dataset')


parser.add_argument('--seed', type=int, default=42)


