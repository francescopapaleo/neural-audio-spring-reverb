# config.py

from argparse import ArgumentParser

parser = ArgumentParser()

# Add PROGRAM level args
parser.add_argument('--root_dir', type=str, default='./', 
                    help='main folder')

parser.add_argument('--data_dir', type=str, default='./data', 
                    help='default dataset folder')

parser.add_argument('--models_dir', type=str, default='./models_trained', 
                    help='folder to store state_dict after training and load for eval or inference')

parser.add_argument('--results_dir', type=str, default='./results', 
                    help='folder to store results/processed files')

parser.add_argument('--audio_dir', type=str, default='./audio', 
                    help='folder for raw audio')

parser.add_argument('--target_dir', type=str, 
                    help='set target folder for a specific function')

parser.add_argument('--target_file', type=str, 
                    help='set target file for a specific function ' )

parser.add_argument('--sr', type=int, default=16000, 
                    help='sampling rate frequency, default: 16KHz')

parser.add_argument('--save', type=str, 
                    help='save weights and biases as')

parser.add_argument('--load', type=str, 
                    help='load weights and biases from')

parser.add_argument('--device', type=str, default='cpu', 
                    help='set the device')

parser.add_argument('--input', type=str, 
                    help="input file to process")

parser.add_argument('--split', type=str, default='train',
                    help='select test/train split of the dataset')

parser.add_argument('--sample_idx', type=int, default=0, 
                    help='The index of the sample from a dataset')

parser.add_argument('--iters', type=int, default=100)

parser.add_argument('--batch_size', type=int, default=1)

parser.add_argument('--shuffle', type=bool, default=True)

parser.add_argument('--seed', type=int, default=42)
