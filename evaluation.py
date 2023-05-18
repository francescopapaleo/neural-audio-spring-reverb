import torch
import scipy
import numpy as np
import h5py

from tcn_model import TCN, causal_crop
from config import *

# Load the h5 files
with h5py.File('/homedtic/fpapaleo/smc-spring-reverb/dataset_subset/dry_train_subset.h5', 'r') as h5f:
    x_data = h5f['Xtrain_subset'][:]

with h5py.File('/homedtic/fpapaleo/smc-spring-reverb/dataset_subset/wet_train_subset.h5', 'r') as h5f:
    y_data = h5f['Ytrain_0_subset'][:]

# # Convert the data to PyTorch tensors
# x_tensor = torch.tensor(x_data, dtype=torch.float32)
# y_tensor = torch.tensor(y_data, dtype=torch.float32)

# # Concatenate the tensors along the time (second) dimension
# x_concat = torch.cat(tuple(x_tensor), dim=1)
# y_concat = torch.cat(tuple(y_tensor), dim=1)

# # Check the shape of the concatenated tensors
# print("Concatenated x shape:", x_concat.shape)
# print("Concatenated y shape:", y_concat.shape)

x_concatenated = np.concatenate(x_data, axis=0)
y_concatenated = np.concatenate(y_data, axis=0)

x_torch = torch.tensor(x_concatenated, dtype=torch.float32)
y_torch = torch.tensor(y_concatenated, dtype=torch.float32)

x = x_torch
y = y_torch

if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

# Reshape the audio
x_batch = x.view(1, 1, -1)
y_batch = y.view(1, 1, -1)
c = torch.tensor([0.0, 0.0], device=device).view(1,1,-1)

_, x_ch, x_samp = x_batch.size()
_, y_ch, y_samp = y_batch.size()

# Load the model
# Instantiate the model with the same parameters as before
model = TCN(
    n_inputs=input_channels,
    n_outputs=output_channels,
    cond_dim=cond_dim,
    kernel_size=kernel_size,
    n_blocks=n_blocks,
    dilation_growth=dilation_growth,
    n_channels=n_channels).float()

rf = model.compute_receptive_field()

# Load the model weights
model.load_state_dict(torch.load('model_weights.pth'))

model.eval()
x_pad = torch.nn.functional.pad(x_batch, (rf-1, 0))
with torch.no_grad():
  y_hat = model(x_pad, c)

input = causal_crop(x_batch.view(-1).detach().cpu().numpy(), y_hat.shape[-1])
output = y_hat.view(-1).detach().cpu().numpy()
target = causal_crop(y_batch.view(-1).detach().cpu().numpy(), y_hat.shape[-1])

# apply highpass to output
sos = scipy.signal.butter(8, 20.0, fs=sample_rate, output="sos", btype="highpass")
output = scipy.signal.sosfilt(sos, output)

input /= np.max(np.abs(input))
output /= np.max(np.abs(output))
target /= np.max(np.abs(target))

# Calculate Mean Squared Error (MSE)
mse = np.mean((output - target) ** 2)

# Print or use the MSE value as needed
print(f"MSE: {mse}")