from config import *

import torch
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from model import TCN, causal_crop
from dataload import PlateSpringDataset
from utils.plot import plot_compare_waveform, plot_zoom_waveform, get_spectrogram, plot_compare_spectrogram

def predict(model_file, data_dir):
    print("## Inference started...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    test_dataset = PlateSpringDataset(data_dir, split='test')

    x_test = test_dataset.concatenate_samples(test_dataset.dry_data)
    y_test = test_dataset.concatenate_samples(test_dataset.wet_data)

    x = torch.tensor(x_test, dtype=torch.float32)
    y = torch.tensor(y_test, dtype=torch.float32)
    c = torch.tensor([0.0, 0.0], device=device).view(1,1,-1)

    # Load the model
    model = TCN(
        n_inputs=INPUT_CH,
        n_outputs=OUTPUT_CH,
        cond_dim=model_params["cond_dim"], 
        kernel_size=model_params["kernel_size"], 
        n_blocks=model_params["n_blocks"], 
        dilation_growth=model_params["dilation_growth"], 
        n_channels=model_params["n_channels"])

    # Load the state dictionary
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    rf = model.compute_receptive_field()
    x_pad = torch.nn.functional.pad(x, (rf-1, 0))

    with torch.no_grad():
        y_hat = model(x_pad, c)

    mse = torch.nn.MSELoss()
    mse = mse(y_hat, y)

    error_mean = torch.mean((y_hat - y) ** 2)
    signal_mean = torch.mean(y ** 2)
    esr_mean = error_mean / (signal_mean + 1e-10)

    mse_sum = torch.nn.MSELoss(reduction='sum')
    error_sum = mse_sum(y_hat, y)
    signal_sum = mse_sum(y, torch.zeros_like(y))
    esr_sum = error_sum / (signal_sum + 1e-10)

    print(str(model_file))
    print(f"Mean Squared Error: {mse}")
    print(f"Error-to-Signal Ratio (mean): {esr_mean}")
    print(f"Error-to-Signal Ratio (sum): {esr_sum}")
    print("")

    error = torch.sum(torch.pow(y_hat - y, 2))
    signal = torch.sum(torch.pow(y, 2))
    esr = error / (signal + 1e-10)
    print(f"Error-to-Signal Ratio: {esr}")

    print("Saving audio files")
    torchaudio.save(os.path.join(RESULTS, "x.wav"), x, SAMPLE_RATE)
    torchaudio.save(os.path.join(RESULTS, "y_pred.wav"), y_hat, SAMPLE_RATE)
    torchaudio.save(os.path.join(RESULTS, "y_.wav"), y, SAMPLE_RATE)

    print("Plotting the results")
    plot_compare_waveform(y[0], y_hat[0], SAMPLE_RATE)
    plot_zoom_waveform(y[0], y_hat[0], SAMPLE_RATE, 0.5, 0.6)

    y_spec = get_spectrogram(y)
    y_pred_spec = get_spectrogram(y_hat)
    x_spec = get_spectrogram(x)

    plot_compare_spectrogram(y_spec, y_pred_spec, x_spec, titles=["Spectrogram of y", "Spectrogram of y_pred", "Spectrogram of x"])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_file", type=str,default=MODEL_FILE, help="Path to the model file")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the data directory")
    args = parser.parse_args()

    predict(args.model_file, args.data_dir)