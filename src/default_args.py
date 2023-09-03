import os
import glob
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='')

    # Data
    parser.add_argument('--dataset', type=str, default='egfxset', help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to the data folder')

    # Paths
    parser.add_argument('--audio_dir', type=str, default='audio', help='Path (rel) to the audio files')
    parser.add_argument('--log_dir', type=str, default='logs', help='Path (rel) to  the log directory')
    parser.add_argument('--results_dir', type=str, default='results', help='Path (rel) to the results directory')
    parser.add_argument('--plots_dir', type=str, default='results/plots', help='Path (rel) to the plot directory')
    parser.add_argument('--models_dir', type=str, default='models', help='Path (rel) to models checkpoints directory')

    # Audio
    parser.add_argument('--sample_rate', type=int, default=48000, help='sample rate of the audio')
    parser.add_argument('--bit_rate', type=int, default=24, help='bits per second')

    # Model
    parser.add_argument('--conf', type=str, default=None, help='The configuration to use')
    parser.add_argument('--device', type=str, default=None, help='Set device to run the model on')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path (rel) to checkpoint file')

    # Training
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='maximum number of epochs for training')
    parser.add_argument('--pre_emphasis', type=float, default=None, help='pre-emphasis filter coefficient')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--early_stop', type=int, default=None, help='number of epochs to wait before early stopping')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs to wait before reducing lr')

    parser.add_argument('--input', type=str, default=None, help='Path (rel) to audio file to process')
    parser.add_argument('--mix', type=float, default=100.0, help='mix parameter for the model')

    # Measurements
    parser.add_argument("--duration", type=float, default=5.0, help="duration in seconds")

    return parser.parse_args()

