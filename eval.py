import os
import sys
import numpy as np
import torch
import torch.nn as nn
import auraloss
import pyloudnorm as pyln

from torch.utils.data import DataLoader
from dataload import PlateSpringDataset 
from argparse import ArgumentParser
from model import TCN
from config import *



def evaluate_model(model_file, data_dir, batch_size, sample_rate):
    print("## Test started...")

    # Load the subset
  
    dataset = PlateSpringDataset(data_dir, split='test')
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    dataiter = iter(test_loader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}") 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_ch = batch_size
    y_ch = batch_size

    # Instantiate the model
    model = TCN(
        n_inputs=x_ch,
        n_outputs=y_ch,
        cond_dim=model_params["cond_dim"], 
        kernel_size=model_params["kernel_size"], 
        n_blocks=model_params["n_blocks"], 
        dilation_growth=model_params["dilation_growth"], 
        n_channels=model_params["n_channels"])
    
    # Load the trained model to evaluate
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    # Metrics
    results = {
    "l1_loss": [],
    "stft_loss": [],
    "lufs_diff": [],
    "aggregate_loss": []
    }

    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    stft = auraloss.freq.STFTLoss()
    meter = pyln.Meter(44100)

    c = torch.tensor([0.0, 0.0], device=device).view(1, 1, -1)

    # Evaluation Loop
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.float(), target.float()
            
            if torch.cuda.is_available():
                input = input.to(device)
                target = target.to(device)
                
            rf = model.compute_receptive_field()
            input_pad = torch.nn.functional.pad(input, (rf-1, 0))

            output = model(input_pad)

            # Calculate the metrics
            mse_loss = mse(output, target).cpu().numpy()
            l1_loss = l1(output, target).cpu().numpy()      
            stft_loss = stft(output, target).cpu().numpy()
            aggregate_loss = l1_loss + stft_loss 

            target_lufs = meter.integrated_loudness(target.squeeze().cpu().numpy())
            output_lufs = meter.integrated_loudness(output.squeeze().cpu().numpy())
            lufs_diff = np.abs(output_lufs - target_lufs)
        
            results["l1_loss"].append(l1_loss)
            results["stft_loss"].append(stft_loss)
            results["lufs_diff"].append(lufs_diff)
            results["aggregate_loss"].append(aggregate_loss)
        
            # print(f"Batch {i} - L1: {l1_loss} - STFT: {stft_loss} - LUFS: {lufs_diff} - Agg: {aggregate_loss}")

    print(f"Average L1 loss: {np.mean(results['l1_loss'])}")
    print(f"Average STFT loss: {np.mean(results['stft_loss'])}")
    print(f"Average LUFS difference: {np.mean(results['lufs_diff'])}")
    print(f"Average Aggregate Loss: {np.mean(results['aggregate_loss'])}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default=MODEL_FILE)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=model_params["batch_size"])
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE)
    args = parser.parse_args()

    evaluate_model(args.model_file, args.data_dir, args.batch_size, args.sample_rate)