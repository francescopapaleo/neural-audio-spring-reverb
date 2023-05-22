import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import torchsummary
import auraloss
from matplotlib import pyplot as plt
import numpy as np

from dataload import PlateSpringDataset
from tcn import TCN, causal_crop
from utils.plot import plot_compare_waveform, plot_zoom_waveform

torch.backends.cudnn.benchmark = True

model_params = {
    "cond_dim": 0,
    "n_blocks": 5,
    "dilation_growth": 10,
    "kernel_size": 6,
    "n_channels": 32,
    "length": 88800,
    "lr": 0.001,
    "batch_size": 1,
    "c": 0.0,
    'in_ch': 1,
    'out_ch': 1,
    "gain_dB": -0.1,
    "c0": 0.6,
    "c1": 0,
    "mix": 100,
    "width": 21,
    "max_length": 30,
    "stereo": False,
    "tail": True
    }

parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--main_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--models_dir', type=str, default='./models')
parser.add_argument('--filename', type=str, default='tcn_weights.pth')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--test_subset', type=str, default='test')
parser.add_argument('--iters', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--model_params', type=dict, default=model_params)

args = parser.parse_args()
sample_rate = args.sample_rate
loss_tracker = []

print("## Loading data...")
train_dataset = PlateSpringDataset(args.data_dir, split=args.train_subset)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=args.shuffle)

x = train_dataset.concatenate_samples(train_dataset.dry_data)
y = train_dataset.concatenate_samples(train_dataset.wet_data)

print("## Training...")
# Load tensors
x = torch.tensor(x, dtype=torch.float32).unsqueeze_(0)
y = torch.tensor(y, dtype=torch.float32).unsqueeze_(0)
c = torch.tensor([0.0, 0.0]).view(1,x.shape[0],-1)
x = x.view(1,x.shape[0],-1)
y = y.view(1,y.shape[0],-1)
    
print(f"Shape of x {x.shape} - Shape of y {y.shape} - Shape of c: {c.shape}")

# crop length
x_batch = x[:,0:1,:]
y_batch = y[:,0:1,:]
x_ch = x_batch.size(0)
y_ch = y_batch.size(0)

model = TCN(
    n_inputs=x_ch,
    n_outputs=y_ch,
    cond_dim=model_params["cond_dim"], 
    kernel_size=model_params["kernel_size"], 
    n_blocks=model_params["n_blocks"], 
    dilation_growth=model_params["dilation_growth"], 
    n_channels=model_params["n_channels"],
    )

# Receptive field and number of parameters
rf = model.compute_receptive_field()
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Parameters: {params*1e-3:0.3f} k")
print(f"Receptive field: {rf} samples or {(rf/ sample_rate)*1e3:0.1f} ms")

# Loss function
loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[32, 128, 512, 2048],
    win_lengths=[32, 128, 512, 2048],
    hop_sizes=[16, 64, 256, 1024])
loss_fn_l1 = torch.nn.L1Loss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), model_params['lr'])
ms1 = int(args.iters * 0.8)
ms2 = int(args.iters * 0.95)
milestones = [ms1, ms2]
print(
    "Learning rate schedule:",
    f"1:{model_params['lr']:0.2e} ->",
    f"{ms1}:{model_params['lr']*0.1:0.2e} ->",
    f"{ms2}:{model_params['lr']*0.01:0.2e}",
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones,
    gamma=0.1,
    verbose=False,
)

device = torch.device(args.device)
model.to(device)
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)
c = c.to(device)
    
torchsummary.summary(model, [(1,65536), (1,2)], device="cpu")

# Training loop
for n in range(args.iters):
    optimizer.zero_grad()   # zero the gradient buffers

    # Crop input and target data
    start_idx = rf
    stop_idx = start_idx + model_params["length"]
    x_crop = x_batch[..., start_idx - rf + 1 : stop_idx]
    y_crop = y_batch[..., start_idx : stop_idx]

    # Forward pass
    y_hat = model(x_crop, c)

    # Compute the loss
    loss = loss_fn(y_hat, y_crop)
    # loss_fn(y_hat, y_crop)

    loss.backward()         # Backward pass
    optimizer.step()        # Update the model parameters
    scheduler.step()        # Update the learning rate scheduler

    if (n + 1) % 1 == 0:
            loss_info = f"Loss at iteration {n+1}: {loss.item():0.3e}"
            print(f" {loss_info} | ")
            loss_tracker.append(loss.item())

    y_hat /= y_hat.abs().max()

# Plot the results
# plot_compare_waveform(y_crop, y_hat, sample_rate)
# plot_zoom_waveform(y_crop, y_hat, t_start=0.0, t_end=0.1, sample_rate)
# plt.plot(np.array(loss_tracker), label='loss')
# plt.show()

save_to = Path(args.models_dir) / args.filename
torch.save(model.state_dict(), save_to, 
           _use_new_zipfile_serialization=False)
