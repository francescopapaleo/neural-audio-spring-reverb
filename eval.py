from pathlib import Path
from argparse import ArgumentParser

from main import model_params, fs, AUDIO_DIR, DATA_DIR, RESULTS
from argparse import ArgumentParser

import numpy as np
import torch
import torchsummary
import auraloss
import pyloudnorm as pyln

from torch.utils.data import DataLoader
from scipy.io import wavfile
from dataload import PlateSpringDataset 
from tcn import TCN
from utils.plot import plot_compare_waveform, plot_zoom_waveform
import matplotlib.pyplot as plt

import pytorch_lightning as pl
torch.backends.cudnn.benchmark = True

pytorch_total_params = sum(p.numel() for p in model.parameters())

# Define your constants here before using them in the ArgumentParse
data_dir = DATA_DIR  # or provide your own default value
results_path = RESULTS  # or provide your own default value
batch_size = model_params["batch_size"]
fs = fs

# Initialize lists for storing metric values
mse_loss_values = []
l1_loss_values = []
stft_loss_values = []
lufs_diff_values = []
aggregate_loss_values = []

def evaluate_model(filename, data_dir, batch_size, fs):
    print("## Test started...")

    # Load the subset  
    dataset = PlateSpringDataset(data_dir, split='test')
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    dataiter = iter(test_loader)

    print(torch.__version__)
    dev = torch.device('cpu')
    # dev = torch.device('cuda')
    print(f"Using device: {dev}")

    # x_ch = batch_size
    # y_ch = batch_size

    # Instantiate the model
    model = TCN(
        n_inputs=model_params["in_ch"],
        n_outputs=model_params["out_ch"],
        cond_dim=model_params["cond_dim"], 
        kernel_size=model_params["kernel_size"], 
        n_blocks=model_params["n_blocks"], 
        dilation_growth=model_params["dilation_growth"], 
        n_channels=model_params["n_channels"]
        )
    # model.load_state_dict(torch.load(filename))
    model = torch.load(filename)
    model.eval()

    print(f'Type of loaded model: {type(model)}')

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
    meter = pyln.Meter(fs)

    c = torch.tensor([0.0, 0.0]).view(1, 1, -1)

    # Evaluation Loop
    with torch.no_grad():
        for input, target in test_loader:
            input, target = input.float(), target.float()
            
            if torch.cuda.is_available():
                input = input.to(dev)
                target = target.to(dev)
                
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

        print(f"Average L1 loss: {np.mean(results['l1_loss'])}")
        print(f"Average STFT loss: {np.mean(results['stft_loss'])}")
        print(f"Average LUFS difference: {np.mean(results['lufs_diff'])}")
        print(f"Average Aggregate Loss: {np.mean(results['aggregate_loss'])}")
        
        # Store metric values over time
        l1_loss_values.append(l1_loss)
        stft_loss_values.append(stft_loss)
        lufs_diff_values.append(lufs_diff)
        aggregate_loss_values.append(aggregate_loss)

        # Plotting the metrics over time
        time_values = range(len(l1_loss_values))

        plt.figure(figsize=(12, 6))
        plt.plot(time_values, l1_loss_values, label="L1 Loss")
        plt.plot(time_values, stft_loss_values, label="STFT Loss")
        plt.plot(time_values, lufs_diff_values, label="LUFS Difference")
        plt.plot(time_values, aggregate_loss_values, label="Aggregate Loss")
        plt.xlabel("Time")
        plt.ylabel("Metric Value")
        plt.title("Metrics Over Time")
        plt.legend()
        plt.show()


    output = output.view(1, -1)
    target = target.view(1, -1)
    input = input.view(1, -1)
    
    output = output.squeeze().numpy()
    target = target.squeeze().numpy()
    input = input.squeeze().numpy()

    # print('Saving audio files...')
    out_path = Path(results / 'eval_output.wav')
    wavfile.write(out_path, fs, output)
    
    target_path = Path(results / 'eval_target.wav')
    wavfile.write(target_path, fs, target)
    
    in_path = Path(results / 'eval_input.wav')
    wavfile.write(in_path, fs, input)

    print('Saving plots...')
    plot_compare_waveform(target, output)
    plot_zoom_waveform(target, output, t_start=0.5, t_end=0.6)
    



evaluate_model(tcn_1000.pth, 'data', 1, 16000)
