import torch
import auraloss
import numpy as np
import h5py
import scipy

from tqdm import tqdm

from torch.utils.data import DataLoader
from dataloader_subset import CustomH5Dataset, load_data
from tcn_model import TCN, causal_crop
from config import *

# Load the dataset
# train_dry_data_list, train_wet_data_list, val_dry_data_list, val_wet_data_list = load_data(DATASET)

# train_dataset = CustomH5Dataset(train_dry_data_list, train_wet_data_list)
# val_dataset = CustomH5Dataset(val_dry_data_list, val_wet_data_list)

# train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
# val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

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

print("Concatenated x shape:", x_torch.shape)
print("Concatenated y shape:", y_torch.shape)

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


# Instantiate the model
model = TCN(
    n_inputs=input_channels,
    n_outputs=output_channels,
    cond_dim=cond_dim, 
    kernel_size=kernel_size, 
    n_blocks=n_blocks, 
    dilation_growth=dilation_growth, 
    n_channels=n_channels)

rf = model.compute_receptive_field()
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Loss function
loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[32, 128, 512, 2048],
    win_lengths=[32, 128, 512, 2048],
    hop_sizes=[16, 64, 256, 1024])
loss_fn_l1 = torch.nn.L1Loss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr)
ms1 = int(n_iters * 0.8)
ms2 = int(n_iters * 0.95)
milestones = [ms1, ms2]
print(
    "Learning rate schedule:",
    f"1:{lr:0.2e} ->",
    f"{ms1}:{lr*0.1:0.2e} ->",
    f"{ms2}:{lr*0.01:0.2e}",
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones,
    gamma=0.1,
    verbose=False,
)

# Move tensors to GPU
if torch.cuda.is_available():
    model.to(device)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    c = c.to(device)

# pad input so that output is same size as input
#x_pad = torch.nn.functional.pad(x_batch, (rf-1, 0))

# Training loop
pbar = tqdm(range(n_iters))
for n in pbar:
    # Zero gradients
    optimizer.zero_grad()

    # Crop input and target data
    start_idx = rf
    stop_idx = start_idx + length
    x_crop = x_batch[..., start_idx - rf + 1 : stop_idx]
    y_crop = y_batch[..., start_idx : stop_idx]

    # Forward pass
    y_hat = model(x_crop, c)

    # Compute the loss
    loss = loss_fn(y_hat, y_crop)  # + loss_fn_l1(y_hat, y_crop)

    # Backward pass
    loss.backward()

    # Update the model parameters
    optimizer.step()

    # Update the learning rate scheduler
    scheduler.step()

    if (n + 1) % 1 == 0:
      pbar.set_description(f" Loss: {loss.item():0.3e} | ")

y_hat /= y_hat.abs().max()

torch.save(model.state_dict(), 'model_weights.pth')

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