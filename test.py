import torch
import scipy
import numpy as np

from model import TCN, causal_crop
from dataloader import SubsetRetriever
from config import *

import torch.nn as nn
import matplotlib.pyplot as plt

# Use GPU if available
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")

# Expects data is a 2D array of shape (n_channels, n_samples)

# Load the subset
subset_retriever = SubsetRetriever('dataset_subset')
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

model.load_state_dict(torch.load('model_TCN_weights.pth'))
model.eval()

# Receptive field
rf = model.compute_receptive_field()

# Pad the input signal
x_pad = torch.nn.functional.pad(x, (rf-1, 0))

with torch.no_grad():
  y_pred = model(x_pad, c)

# input = causal_crop(x_pad.view(-1).detach().cpu().numpy(), y_pred.shape[-1])
# output = y_pred.view(-1).detach().cpu().numpy()
# target = causal_crop(y.view(-1).detach().cpu().numpy(), y_pred.shape[-1])

# apply highpass to output
# sos = scipy.signal.butter(8, 20.0, fs=sample_rate, output="sos", btype="highpass")
# output = scipy.signal.sosfilt(sos, output)

# input /= np.max(np.abs(input))
# output /= np.max(np.abs(output))
# target /= np.max(np.abs(target))

print(f"Ground Truth: {y.shape}")
print(f"Prediction: {y_pred.shape}")

# Pad output and target to match
# max_lex = max(y_pred[0], y[0])
# y = torch.nn.functional.pad(y[0], (0, max_lex - y[0]))
# y_pred = torch.nn.functional.pad(y_pred[0], (0, max_lex - y_pred[0]))

# Define the MSE loss function
mse = torch.nn.MSELoss()
metric = mse(y_pred, y)

print(f"MSE: {metric.item()}")

# Plot the data
plt.figure(figsize=(12, 6))

plt.plot(input, label='Input')
plt.plot(output, label='Output', linestyle='--')
plt.plot(target, label='Target', linestyle=':')

plt.legend()
plt.grid(True)
plt.title('Model Evaluation')
plt.xlabel('Samples')
plt.ylabel('Value')

plt.show()
