import torch
import torch.nn as nn
import torch.optim as optim
import scipy.signal
import auraloss
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader
from dataloader_subset import CustomH5Dataset, load_data
from model import TCN, causal_crop
from config import *


# Load the dataset
train_dry_data_list, train_wet_data_list, val_dry_data_list, val_wet_data_list = load_data(DATASET)

train_dataset = CustomH5Dataset(train_dry_data_list, train_wet_data_list)
val_dataset = CustomH5Dataset(val_dry_data_list, val_wet_data_list)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

# Define input and output channels
x_ch = 1
y_ch = 1

if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

# Get the shapes of the first batch of data
first_input_batch, first_output_batch = next(iter(train_dataloader))
input_channels = first_input_batch.size(1)
output_channels = first_output_batch.size(1)

# Instantiate the model
model = TCN(
    n_inputs=input_channels,
    n_outputs=output_channels,
    cond_dim=cond_dim,
    kernel_size=kernel_size,
    n_blocks=n_blocks,
    dilation_growth=dilation_growth,
    n_channels=n_channels).float()

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

# Initialize progress bar
pbar = tqdm(range(n_iters))

# Move the model to GPU if available
if torch.cuda.is_available():
    model.to(device)

# Training loop
for n in pbar:
    # Get next batch from DataLoader
    x_batch, y_batch = next(iter(train_dataloader))
    x_batch = x_batch.float().to(device)  # Convert to float32 and move to the device
    y_batch = y_batch.to(device)

    # Define the conditioning tensor
    c = torch.tensor([0.0, 0.0], device=device, dtype=torch.float32).view(1, 1, -1)

    # Zero gradients
    optimizer.zero_grad()

    # Define start and stop indices for cropping
    start_idx = rf
    stop_idx = start_idx + length

    # Crop input and output
    x_crop = x_batch[..., start_idx - rf + 1 : stop_idx]
    y_crop = y_batch[..., start_idx : stop_idx]

    # Make prediction
    y_hat = model(x_crop, c)

    # Calculate loss
    loss = loss_fn(y_hat, y_crop)

    # Update gradients
    loss.backward()

    # Perform optimization step
    optimizer.step()

    # Update learning rate scheduler
    scheduler.step()

    # Update progress bar description
    if (n + 1) % 1 == 0:
        pbar.set_description(f" Loss: {loss.item():0.3e} | ")

# Normalize output
y_hat /= y_hat.abs().max()

torch.save(model.state_dict(), 'model_weights.pth')