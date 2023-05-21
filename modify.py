import json
from pathlib import Path
from argparse import ArgumentParser


import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
import auraloss
import pyloudnorm as pyln

from model import TCN, causal_crop, center_crop
from dataload import PlateSpringDataset
from sandbox.config import *
from argparse import ArgumentParser


def test_model(model_file, data_dir, n_parts, sample_rate):
    print("## Evaluation started...")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
        
    print("")
    print(f"Name: {model_file}")

    # Load the subset
    test_dataset = PlateSpringDataset(data_dir, split='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # subset_dry, subset_wet, indices = dataset.load_random_subset(9, seed)
    x = test_dataset.concatenate_samples(test_dataset.dry_data)
    y = test_dataset.concatenate_samples(test_dataset.wet_data)

    # Load tensors
    input = torch.tensor(x, dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)

    # Add channel dimension
    x = x[:, None, :]
    y = y[:, None, :]
    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    c = torch.tensor([0.0, 0.0], device=device).view(1,1,-1)

    # Instantiate the model
    model = TCN(
        n_inputs=INPUT_CH,
        n_outputs=OUTPUT_CH,
        cond_dim=model_params["cond_dim"], 
        kernel_size=model_params["kernel_size"], 
        n_blocks=model_params["n_blocks"], 
        dilation_growth=model_params["dilation_growth"], 
        n_channels=model_params["n_channels"])

    # Load the trained model to evaluate
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    model_id = os.path.splitext(os.path.basename(model_file))[0]

    # Move tensors to GPU
    if torch.cuda.is_available():
        model.to(device)
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)

    # set up loss functions for evaluation
    l1 = nn.L1Loss()
    mse = nn.MSELoss()
    
    results = {}  # Initialize the results dictionary
    overall_results = {}  # Initialize the overall results dictionary

    # # Creating n chunks
    # x_parts = torch.chunk(x, n_parts)
    # y_parts = torch.chunk(y, n_parts)

    total_mse = 0
    total_esr = 0
    total_l1 = 0
    total_stft = 0
    total_lufs = 0

    for i, data in enumerate(test_dataloader):
        
        input, target = data

        with torch.no_grad():
            output = model(input, c)

        print(f'Input shape: {input.shape}')
        print(f'Target shape: {target.shape}')
        print(f'Output shape: {output.shape}')

        input_crop = causal_crop(input, output.shape[-1])
        target_crop = causal_crop(target, output.shape[-1])
        
        print(f'Input crop shape: {input_crop.shape}')
        print(f'Target crop shape: {target_crop.shape}')

        for idx, (i, o, t, c) in enumerate(zip(
                                            torch.split(input_crop, 1, dim=0),
                                            torch.split(output, 1, dim=0),
                                            torch.split(target_crop, 1, dim=0),
                                            torch.split(c, 1, dim=0))):
            
            l1_loss = l1(o, t).cpu().numpy()
            stft_loss = stft(o, t).cpu().numpy()
            aggregate_loss = l1_loss + stft_loss 

            target_lufs = meter.integrated_loudness(t.squeeze().cpu().numpy())
            output_lufs = meter.integrated_loudness(o.squeeze().cpu().numpy())
            l1_lufs = np.abs(output_lufs - target_lufs)

            l1i_loss = (l1(i, t) - l1(o, t)).cpu().numpy()
            stfti_loss = (stft(i, t) - stft(o, t)).cpu().numpy()

            params = c.squeeze().cpu().numpy()
            params_key = f"{params[0]:1.0f}-{params[1]*100:03.0f}"

        if params_key not in list(results.keys()):
                results[params_key] = {
                    "L1" : [l1_loss],
                    "L1i" : [l1i_loss],
                    "STFT" : [stft_loss],
                    "STFTi" : [stfti_loss],
                    "LUFS" : [l1_lufs],
                    "Agg" : [aggregate_loss]
                }
        else:
            results[params_key]["L1"].append(l1_loss)
            results[params_key]["L1i"].append(l1i_loss)
            results[params_key]["STFT"].append(stft_loss)
            results[params_key]["STFTi"].append(stfti_loss)
            results[params_key]["LUFS"].append(l1_lufs)
            results[params_key]["Agg"].append(aggregate_loss)

    # store in dict
    l1_scores = []
    lufs_scores = []
    stft_scores = []
    agg_scores = []
    print("-" * 64)
    print("Config      L1         STFT      LUFS")
    print("-" * 64)
    for key, val in results.items():
        print(f"{key}    {np.mean(val['L1']):0.2e}    {np.mean(val['STFT']):0.3f}       {np.mean(val['LUFS']):0.3f}")

        l1_scores += val["L1"]
        stft_scores += val["STFT"]
        lufs_scores += val["LUFS"]
        agg_scores += val["Agg"]

    print("-" * 64)
    print(f"Mean     {np.mean(l1_scores):0.2e}    {np.mean(stft_scores):0.3f}      {np.mean(lufs_scores):0.3f}")
    print()
    overall_results[model_id] = results

# pickle.dump(overall_results, open(f"test_results_{args.eval_subset}.p", "wb" ))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str, default=MODEL_FILE, help="Path to the model file")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the data directory")
    parser.add_argument("--n_parts", type=int, default=10, help="Number of chunks to split the data into")
    parser.add_argument("--sample_rate", type=int, default=SAMPLE_RATE, help="Sample rate of the audio")
    parser.add_argument("--eval_subset", type=str, required=None, help="The evaluation subset")
    parser.add_argument("--save_dir", type=str, help="Directory to save the output, input and target files")
    args = parser.parse_args()

    test_model(args.model_file, args.data_dir, args.n_parts, args.sample_rate)
