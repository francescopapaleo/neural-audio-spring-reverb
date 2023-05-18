import torch
import auraloss
import numpy as np
import h5py
import scipy

from tqdm import tqdm

from dataloader_subset import DataLoader
from model_tcn import TCN, causal_crop
from config import *

# Load the data
data_loader = DataLoader(DATASET)
x_train, y_train, x_valid, y_valid, _ , _ = data_loader.load_data('subset')

# print(x_train.shape)
# print(y_train.shape)
# print(x_valid.shape)
# print(y_valid.shape)


# Concatenate
x_concatenated = np.concatenate(x_train, axis=1)
y_concatenated = np.concatenate(y_train, axis=1)
################################################
# # Concatenate audio data from the first five samples in samples_list
# x_concat = np.concatenate([Xtrain[i, :, 0] for i in samples_list[:5]])
# y_concat = np.concatenate([Ytrain_0[i, :, 0] for i in samples_list[:5]])


# ./data/Xtrain.wav torch.Size([1, 35904000])
# Convert to torch tensors
x_torch = torch.tensor(x_concatenated, dtype=torch.float32)
y_torch = torch.tensor(y_concatenated, dtype=torch.float32)

x = x_torch
y = y_torch

print(x.shape)
print(y.shape)

# Use GPU if available
if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")


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

# Receptive field and number of parameters
rf = model.compute_receptive_field()
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Parameters: {params*1e-3:0.3f} k")
print(f"Receptive field: {rf} samples or {(rf/sample_rate)*1e3:0.1f} ms")

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