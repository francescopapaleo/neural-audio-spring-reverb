# train.py
import numpy as np
import torch
import torchsummary
import auraloss
import pickle
from pathlib import Path
from matplotlib import pyplot as plt

from utils.dataload import PlateSpringDataset
from tcn import TCN, model_params
from utils.plot import plot_compare_waveform, plot_zoom_waveform, plot_loss_function

from config import parser
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)

print("## Loading data...")
train_dataset = PlateSpringDataset(args.data_dir, split=args.split)
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
print(f"Receptive field: {rf} samples or {(rf/ args.sr)*1e3:0.1f} ms")

# Initialize lists for storing losses values
mse_loss_values = []
l1_loss_values = []
stft_loss_values = []
esr_loss_values = []
dc_loss_values = []
snr_loss_values = []

mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
stft = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[32, 128, 512, 2048],
    win_lengths=[32, 128, 512, 2048],
    hop_sizes=[16, 64, 256, 1024])
esr = auraloss.time.ESRLoss()
dc = auraloss.time.DCLoss()
snr = auraloss.time.SNRLoss()

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
    
torchsummary.summary(model, [(1,65536), (1,2)], device=args.device)

# Training loop
for n in range(args.iters):

    optimizer.zero_grad()   # zero the gradient buffers

    # Crop input and target data
    start_idx = rf
    stop_idx = start_idx + model_params["length"]
    x_crop = x_batch[..., start_idx - rf + 1 : stop_idx]
    y_crop = y_batch[..., start_idx : stop_idx]
    input = x_crop
    target = y_crop

    # Forward pass
    output = model(input, c)

    mse_loss = mse(output, target)
    l1_loss = l1(output, target)      
    stft_loss = stft(output, target)
    esr_loss = esr(output, target)
    dc_loss = dc(output, target)
    snr_loss = snr(output, target)
    
    # Compute the loss
    stft_loss.backward()         # Backward pass
    optimizer.step()        # Update the model parameters
    scheduler.step()        # Update the learning rate scheduler

    if (n + 1) % 1 == 0:
            loss_info = f"Epoch {n+1} Training Loss: {stft_loss.item():0.3e}"
            print(f" {loss_info} | ")
            print("")
            print(f"MSE loss: {mse_loss.item():0.3e}")
            print(f"L1 loss: {l1_loss.item():0.3e}")
            print(f"STFT loss: {stft_loss.item():0.3e}")
            print(f"ESR loss: {esr_loss.item():0.3e}")
            print(f"DC loss: {dc_loss.item():0.3e}")
            print(f"SNR loss: {snr_loss.item():0.3e}")


            # Store metric values over time
            mse_loss_values.append(mse_loss.item())
            l1_loss_values.append(l1_loss.item())
            stft_loss_values.append(stft_loss.item())
            esr_loss_values.append(esr_loss.item())
            dc_loss_values.append(dc_loss.item())
            snr_loss_values.append(snr_loss.item())

# Convert lists to numpy arrays for easier manipulation
mse_loss_values = np.array(mse_loss_values)
l1_loss_values = np.array(l1_loss_values)
stft_loss_values = np.array(stft_loss_values)
esr_loss_values = np.array(esr_loss_values)
dc_loss_values = np.array(dc_loss_values)
snr_loss_values = np.array(snr_loss_values)

# Save the loss data
with open('loss_data.pkl', 'wb') as f:
    pickle.dump({
        'mse_loss_values': mse_loss_values,
        'l1_loss_values': l1_loss_values,
        'stft_loss_values': stft_loss_values,
        'esr_loss_values': esr_loss_values,
        'dc_loss_values': dc_loss_values,
        'snr_loss_values': snr_loss_values
    }, f)

print('Saving plots...')
normalized_mse_loss = (mse_loss_values - np.min(mse_loss_values)) / (np.max(mse_loss_values) - np.min(mse_loss_values))
normalized_l1_loss = (l1_loss_values - np.min(l1_loss_values)) / (np.max(l1_loss_values) - np.min(l1_loss_values))
normalized_stft_loss = (stft_loss_values - np.min(stft_loss_values)) / (np.max(stft_loss_values) - np.min(stft_loss_values))
normalized_esr_loss = (esr_loss_values - np.min(esr_loss_values)) / (np.max(esr_loss_values) - np.min(esr_loss_values))
normalized_dc_loss = (dc_loss_values - np.min(dc_loss_values)) / (np.max(dc_loss_values) - np.min(dc_loss_values))
normalized_snr_loss = (snr_loss_values - np.min(snr_loss_values)) / (np.max(snr_loss_values) - np.min(snr_loss_values))

# Plotting the metrics over time
time_values = np.arange(args.iters)  # Create an array of iteration numbers

plt.figure(figsize=(15, 7))
plt.plot(time_values, normalized_mse_loss, label="MSE Loss")
plt.plot(time_values, normalized_l1_loss, label="L1 Loss")
plt.plot(time_values, normalized_stft_loss, label="STFT Loss")
plt.plot(time_values, normalized_esr_loss, label="ESR Loss")
plt.plot(time_values, normalized_dc_loss, label="DC Loss")
plt.plot(time_values, normalized_snr_loss, label="SNR Loss")
plt.xlabel("Epochs")
plt.ylabel("Normalized Metric Value")
plt.title("Training Progress: Metrics Over Epochs")
plt.legend()
plt.savefig(Path(args.results_dir) / 'eval_metrics_plot.png')

# Save the model
save_to = Path(args.models_dir) / args.save
torch.save(model.state_dict(), save_to)

