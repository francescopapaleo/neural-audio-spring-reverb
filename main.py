import argparse
from config import *
from train import train_model
from test import evaluate_model
from pred import predict

def main():
    parser = argparse.ArgumentParser(description='Temporal Convolutional Networks')
    parser.add_argument('--task', type=str, required=True, 
                        choices=['train', 'eval', 'predict'],
                        help='Task to be performed: train, eval or predict')
    args = parser.parse_args()

    task = args.task

    if task == 'train':
        train_model(MODEL_FILE)
    elif task == 'eval':
        evaluate_model(MODEL_FILE) # You will need to define evaluate_model in your evaluation.py
    elif task == 'predict':
        predict(INPUT_FILE)  # You will need to define predict in your predict.py
    else:
        print("Invalid task. Choose from 'train', 'eval' or 'predict'.")

if __name__ == "__main__":
    main()

