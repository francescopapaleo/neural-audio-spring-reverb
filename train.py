# Train the model
from pathlib import Path
from argparse import ArgumentParser
from main import model_params, DATA_DIR, MODELS_DIR, fs
import torch
import auraloss
import os
from tqdm import tqdm

import numpy as np

from dataload import PlateSpringDataset
from model import TCN, causal_crop
from argparse import ArgumentParser
from utils.plot import plot_compare_waveform, plot_zoom_waveform
from matplotlib import pyplot as plt

loss_tracker = []

def train_model(model_file, data_dir):
    print("## Training started...")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the subset
    train_dataset = PlateSpringDataset(data_dir, split='train')

    x = train_dataset.concatenate_samples(train_dataset.dry_data)
    y = train_dataset.concatenate_samples(train_dataset.wet_data)
    
    # Load tensors
    x = torch.tensor(x, dtype=torch.float32).unsqueeze_(0)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze_(0)

    # reshape the audio
    x_batch = x.view(1,x.shape[0],-1)
    y_batch = y.view(1,y.shape[0],-1)
    
    c = torch.tensor([0.0, 0.0], device=device).view(1,x.shape[0],-1)
    
    print(f"Shape of x {x.shape}")
    print(f"Shape of y {y.shape}")
    print(f"Shape of c: {c.shape}")
    
    # crop length
    x_batch = x_batch[:,0:1,:]
    y_batch = y_batch[:,0:1,:]

    x_ch = x_batch.size(0)
    y_ch = y_batch.size(0)

    print(f"Hyperparameters: {model_params}")

    # Instantiate the model
    model = TCN(
        n_inputs=x_ch,
        n_outputs=y_ch,
        cond_dim=model_params["cond_dim"], 
        kernel_size=model_params["kernel_size"], 
        n_blocks=model_params["n_blocks"], 
        dilation_growth=model_params["dilation_growth"], 
        n_channels=model_params["n_channels"])

    # Receptive field and number of parameters
    rf = model.compute_receptive_field()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {params*1e-3:0.3f} k")
    print(f"Receptive field: {rf} samples or {(rf/ fs)*1e3:0.1f} ms")

    # Loss function
    loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048],
        win_lengths=[32, 128, 512, 2048],
        hop_sizes=[16, 64, 256, 1024])
    loss_fn_l1 = torch.nn.L1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), model_params['lr'])
    ms1 = int(model_params['n_iters'] * 0.8)
    ms2 = int(model_params['n_iters'] * 0.95)
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

    # Move tensors to GPU
    if torch.cuda.is_available():
        model.to(device)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        c = c.to(device)
    
    # Training loop
    pbar = tqdm(range(model_params["n_iters"]))
    for n in pbar:
        
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

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Update the learning rate scheduler
        scheduler.step()

        if (n + 1) % 1 == 0:
            loss_info = f"Loss at iteration {n+1}: {loss.item():0.3e}"
            pbar.set_description(f" {loss_info} | ")
            loss_tracker.append(loss.item())
    
    y_hat /= y_hat.abs().max()

    # Plot the results
    # plot_compare_waveform(y_crop, y_hat, fs=SAMPLE_RATE)
    # plot_zoom_waveform(y_crop, y_hat, t_start=0.0, t_end=0.1, fs=SAMPLE_RATE)
    plt.plot(np.array(loss_tracker), label='loss')
    plt.show()

    # Save the model
    save_path = os.path.join(MODELS_DIR, model_file)
    # torch.save(model.state_dict(), save_path)
    torch.save(model_file, save_path)
    
    print(f"Saved model to {model_file}")
    print("")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        default='model_file',
        help="Name of the model file to store",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help="Path to the data directory",
    )

    args = parser.parse_args()

train_model(args.model_file, args.data_dir)