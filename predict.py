import torch
import numpy as np

from model import TCN, causal_crop
from dataloader import SubsetRetriever
from config import *
from plot import plot_compare_waveform, plot_zoom_waveform, plot_compare_spectrum, get_spectrogram, plot_spectrogram

import torch.nn as nn
import matplotlib.pyplot as plt
import torchaudio

print("")
print("# Predicting on new data")

# Use GPU if available
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")

# Expects data is a 2D array of shape (n_channels, n_samples)

# Load the subset
subset_retriever = SubsetRetriever(SUBSET)
_, _, x_test_concate , y_test_concate  = subset_retriever.retrieve_data(concatenate=True)

# Load tensors
x_torch = torch.tensor(x_test_concate, dtype=torch.float32)
y_torch = torch.tensor(y_test_concate, dtype=torch.float32)
c = torch.tensor([0.0, 0.0], device=device).view(1,1,-1)

x = x_torch
y = y_torch

# Instantiate the model
model = TCN(
    n_inputs=input_channels,
    n_outputs=output_channels,
    cond_dim=cond_dim, 
    kernel_size=kernel_size, 
    n_blocks=n_blocks, 
    dilation_growth=dilation_growth, 
    n_channels=n_channels)

load_this_model = os.path.join(MODELS, model_for_prediction)

model = torch.load(load_this_model)
model.eval()

# Receptive field
rf = model.compute_receptive_field()

# Pad the input signal
x_pad = torch.nn.functional.pad(x, (rf-1, 0))

with torch.no_grad():
    y_pred = model(x_pad, c)
    
# Mean squared error
mse = torch.nn.MSELoss()
mse = mse(y_pred, y)

# Error to signal 
error = torch.sum(torch.pow(y_pred - y_pred, 2))
signal = torch.sum(torch.pow(y, 2))
esr = error / (signal + 1e-10)

print(str(model_to_evaluate))
print(f"Average MSE: {mse}")
print(f"Average ESR: {esr}")
print("")

print("Saving audio files")
torchaudio.save(os.path.join(AUDIO, "x.wav"), x, sample_rate)
torchaudio.save(os.path.join(AUDIO, "y_pred.wav"), y_pred, sample_rate)
torchaudio.save(os.path.join(AUDIO, "y_.wav"), y, sample_rate)

print("Plotting the results")
plot_compare_waveform(y[0], y_pred[0], sample_rate)
plot_zoom_waveform(y[0], y_pred[0], sample_rate, 0.5, 0.6)
# plot_compare_spectrum(y[0], y_pred[0], x[0], sample_rate)

# Let's assume 'y', 'y_pred', and 'x' are your waveforms
# Ensure they are in the format [channel, time], where channel is 1 for mono audio
y_spec = get_spectrogram(y)
y_pred_spec = get_spectrogram(y_pred)
x_spec = get_spectrogram(x)

# Plot the spectrograms
plot_spectrogram(y_spec, title="Spectrogram of y")
plot_spectrogram(y_pred_spec, title="Spectrogram of y_pred")
plot_spectrogram(x_spec, title="Spectrogram of x")
