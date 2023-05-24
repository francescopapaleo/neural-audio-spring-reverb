from config import parser
from pathlib import Path

import torch
import torchaudio
import torchsummary
import auraloss
from matplotlib import pyplot as plt
import numpy as np

from utils.dataload import PlateSpringDataset
from tcn import TCN, causal_crop, model_params
from utils.plot import plot_compare_waveform, plot_zoom_waveform, plot_compare_spectrogram

import pyloudnorm as pyln

torch.backends.cudnn.benchmark = True


args = parser.parse_args()
sample_rate = args.sr

print("## Loading data...")
test_dataset = PlateSpringDataset(args.data_dir, split=args.split)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=args.shuffle)
dataiter = iter(test_dataloader)

x = test_dataset.concatenate_samples(test_dataset.dry_data)
y = test_dataset.concatenate_samples(test_dataset.wet_data)

print("## Loading model...")
device = torch.device(args.device)

load_from = Path(args.models_dir) / args.load

model = TCN(
    n_inputs=1,
    n_outputs=1,
    cond_dim=model_params["cond_dim"], 
    kernel_size=model_params["kernel_size"], 
    n_blocks=model_params["n_blocks"], 
    dilation_growth=model_params["dilation_growth"], 
    n_channels=model_params["n_channels"],
    )
model.load_state_dict(torch.load(load_from))
model = model.to(args.device)  # move the model to the right device
model.eval()  # set the model to evaluation mode

torchsummary.summary(model, [(1,65536), (1,2)], device=args.device)

print("## Testing...")

# Metrics
results = {
"l1_loss": [],
"stft_loss": [],
"lufs_diff": [],
"aggregate_loss": []
}

# Initialize lists for storing metric values
mse_loss_values = []
l1_loss_values = []
stft_loss_values = []
lufs_diff_values = []
aggregate_loss_values = []

mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
stft = auraloss.freq.STFTLoss()
meter = pyln.Meter(sample_rate)

# Evaluation Loop
with torch.no_grad():
    for input, target in test_dataloader:
        input, target = input.float(), target.float()
        
        input = input.to(device)
        target = target.to(device)
            
        rf = model.compute_receptive_field()
        input_pad = torch.nn.functional.pad(input, (rf-1, 0))

        output = model(input_pad)

        # Calculate the metrics
        mse_loss = mse(output, target).cpu().numpy()
        l1_loss = l1(output, target).cpu().numpy()      
        stft_loss = stft(output, target).cpu().numpy()
        aggregate_loss = l1_loss + stft_loss 

        target_lufs = meter.integrated_loudness(target.squeeze().cpu().numpy())
        output_lufs = meter.integrated_loudness(output.squeeze().cpu().numpy())
        lufs_diff = np.abs(output_lufs - target_lufs)
    
        results["l1_loss"].append(l1_loss)
        results["stft_loss"].append(stft_loss)
        results["lufs_diff"].append(lufs_diff)
        results["aggregate_loss"].append(aggregate_loss)
        
        # Store metric values over time
        l1_loss_values.append(l1_loss)
        stft_loss_values.append(stft_loss)
        lufs_diff_values.append(lufs_diff)
        aggregate_loss_values.append(aggregate_loss)


print(f"Average L1 loss: {np.mean(results['l1_loss'])}")
print(f"Average STFT loss: {np.mean(results['stft_loss'])}")
print(f"Average LUFS difference: {np.mean(results['lufs_diff'])}")
print(f"Average Aggregate Loss: {np.mean(results['aggregate_loss'])}")

# print('Saving audio files...')
ofile = Path(args.results_dir) / 'eval_output.wav'
tfile = Path(args.results_dir) / 'eval_target.wav'
ifile = Path(args.results_dir) / 'eval_input.wav'

o_float = output.view(1,-1).cpu().float()
t_float = target.view(1,-1).cpu().float()
i_float = input.view(1,-1).cpu().float()

torchaudio.save(ofile, o_float, sample_rate)
torchaudio.save(tfile, t_float, sample_rate)
torchaudio.save(ifile, i_float, sample_rate)

print('Saving plots...')
# Plotting the metrics over time
time_values = range(len(l1_loss_values))

plt.figure(figsize=(15, 7))
plt.plot(time_values, l1_loss_values, label="L1 Loss")
plt.plot(time_values, stft_loss_values, label="STFT Loss")
plt.plot(time_values, lufs_diff_values, label="LUFS Difference")
plt.plot(time_values, aggregate_loss_values, label="Aggregate Loss")
plt.xlabel("Time")
plt.ylabel("Metric Value")
plt.title("Evaluation: Metrics Over Time (Test Set)")
plt.legend()
plt.savefig(Path(args.results_dir) / 'eval_metrics_plot.png')

o_plot = output.detach().cpu().numpy().reshape(-1)
t_plot = target.detach().cpu().numpy().reshape(-1)
i_plot = input.detach().cpu().numpy().reshape(-1)

plot_compare_waveform(t_plot, o_plot)
plot_zoom_waveform(t_plot, o_plot,args.sr, t_start=0.5, t_end=0.6)
plot_compare_spectrogram(t_float, o_float, i_float,  titles=['target', 'output', 'input'], ylabel="freq_bin", aspect="auto", xmax=None)

