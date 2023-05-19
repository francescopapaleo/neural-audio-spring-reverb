from config import *

import torch
import torchaudio

import numpy as np

from model import TCN, causal_crop
from dataloader import SubsetRetriever
from plot import plot_compare_waveform, plot_zoom_waveform, get_spectrogram, plot_compare_spectrogram

print("")
print("## Inference started...")

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the subset
subset_retriever = SubsetRetriever(SUBSET)
_, _, x_test_concate , y_test_concate  = subset_retriever.retrieve_data(concatenate=True)

# Load tensors (n_channels, n_samples)
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
error_mean = torch.mean((y_pred - y) ** 2)
signal_mean = torch.mean(y ** 2)
esr_mean = error_mean / (signal_mean + 1e-10)

mse_sum = torch.nn.MSELoss(reduction='sum')
error_sum = mse_sum(y_pred, y)
signal_sum = mse_sum(y, torch.zeros_like(y))
esr_sum = error_sum / (signal_sum + 1e-10)

print(str(model_to_evaluate))
print(f"Mean Squared Error: {mse}")
print(f"Error-to-Signal Ratio (mean): {esr_mean}")
print(f"Error-to-Signal Ratio (sum): {esr_sum}")
print("")

# Error to signal 
error = torch.sum(torch.pow(y_pred - y, 2))
signal = torch.sum(torch.pow(y, 2))
esr = error / (signal + 1e-10)

print(f"Error-to-Signal Ratio: {esr}")

print("Saving audio files")
print("")
torchaudio.save(os.path.join(AUDIO, "x.wav"), x, sample_rate)
torchaudio.save(os.path.join(AUDIO, "y_pred.wav"), y_pred, sample_rate)
torchaudio.save(os.path.join(AUDIO, "y_.wav"), y, sample_rate)

print("Plotting the results")
print("")
plot_compare_waveform(y[0], y_pred[0], sample_rate)
plot_zoom_waveform(y[0], y_pred[0], sample_rate, 0.5, 0.6)

# Let's assume 'y', 'y_pred', and 'x' are your waveforms
# Ensure they are in the format [channel, time], where channel is 1 for mono audio
y_spec = get_spectrogram(y)
y_pred_spec = get_spectrogram(y_pred)
x_spec = get_spectrogram(x)

# Plot the spectrograms
plot_compare_spectrogram(y_spec, y_pred_spec, x_spec, titles=["Spectrogram of y", "Spectrogram of y_pred", "Spectrogram of x"])

# Generate an impulse
impulse_duration = 1.0  # You can set this value
impulse_sample_rate = sample_rate  # You can set this value, usually same as the audio sample rate
impulse = generate_impulse(impulse_duration, impulse_sample_rate)

# Feed the impulse to the model
y_pred = feed_model_with_impulse(model, impulse)

# Now you can proceed with the rest of your evaluation as usual, using 'y_pred' as the model's output