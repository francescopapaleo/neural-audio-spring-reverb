from config import parser
from pathlib import Path

import torch
import torchaudio
import torchsummary
import auraloss
import pickle
from matplotlib import pyplot as plt
import numpy as np

from utils.dataload import SpringDataset
from tcn import TCN, causal_crop, model_params
from utils.plot import plot_compare_waveform, plot_zoom_waveform, plot_compare_spectrogram

torch.backends.cudnn.benchmark = True

args = parser.parse_args()
sample_rate = args.sr

print("## Loading data...")
test_set = SpringDataset(args.data_dir, split='test')
dry_subset, wet_subset, indices = test_set.load_random_subset(8)

sampler = torch.utils.data.SubsetRandomSampler(indices)

test_dataloader = torch.utils.data.DataLoader(dataset=test_set, 
                              batch_size=args.batch_size,
                              sampler=sampler)

# Iterate over the test_dataloader and print the batches
for x, y in test_dataloader:
    print("Batch x:")
    print(x)
    print("Batch y:")
    print(y)
    print()  # Print an empty line between batches

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

# Initialize lists for storing metric values
mse_metric_values = []
l1_metric_values = []
stft_metric_values = []
esr_metric_values = []
dc_metric_values = []
snr_metric_values = []

mse = torch.nn.MSELoss()
l1 = torch.nn.L1Loss()
stft = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[32, 128, 512, 2048],
    win_lengths=[32, 128, 512, 2048],
    hop_sizes=[16, 64, 256, 1024])
esr = auraloss.time.ESRLoss()
dc = auraloss.time.DCLoss()
snr = auraloss.time.SNRLoss()

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
        mse_metric = mse(output, target)
        l1_metric = l1(output, target)
        stft_metric = stft(output, target)
        esr_metric = esr(output, target)
        dc_metric = dc(output, target)
        snr_metric = snr(output, target)

        # Store metric values over time
        mse_metric_values.append(mse_metric.item())
        l1_metric_values.append(l1_metric.item())
        stft_metric_values.append(stft_metric.item())
        esr_metric_values.append(esr_metric.item())
        dc_metric_values.append(dc_metric.item())
        snr_metric_values.append(snr_metric.item())

# Convert lists to numpy arrays
mse_metric_values = np.array(mse_metric_values)
l1_metric_values = np.array(l1_metric_values)
stft_metric_values = np.array(stft_metric_values)
esr_metric_values = np.array(esr_metric_values)
dc_metric_values = np.array(dc_metric_values)
snr_metric_values = np.array(snr_metric_values)

# Save metric data
with open(Path(args.results_dir) / 'eval_metrics.pkl', 'wb') as f:
    pickle.dump([
        mse_metric_values, 
        l1_metric_values, 
        stft_metric_values, 
        esr_metric_values, 
        dc_metric_values, 
        snr_metric_values], 
        f)
    
# normalize the metrics
mse_metric_values = (mse_metric_values - np.mean(mse_metric_values)) / np.std(mse_metric_values)
l1_metric_values = (l1_metric_values - np.mean(l1_metric_values)) / np.std(l1_metric_values)
stft_metric_values = (stft_metric_values - np.mean(stft_metric_values)) / np.std(stft_metric_values)
esr_metric_values = (esr_metric_values - np.mean(esr_metric_values)) / np.std(esr_metric_values)
dc_metric_values = (dc_metric_values - np.mean(dc_metric_values)) / np.std(dc_metric_values)
snr_metric_values = (snr_metric_values - np.mean(snr_metric_values)) / np.std(snr_metric_values)

# Print Average Metrics
print('MSE: ', np.mean(mse_metric_values))
print('L1: ', np.mean(l1_metric_values))
print('STFT: ', np.mean(stft_metric_values))
print('ESR: ', np.mean(esr_metric_values))
print('DC: ', np.mean(dc_metric_values))
print('SNR: ', np.mean(snr_metric_values))


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
time_values = np.arange(len(test_dataloader))

plt.figure(figsize=(15, 7))
plt.plot(time_values, mse_metric_values, label="MSE")
plt.plot(time_values, l1_metric_values, label="L1")
plt.plot(time_values, stft_metric_values, label="STFT")
plt.plot(time_values, esr_metric_values, label="ESR")
plt.plot(time_values, dc_metric_values, label="DC")
plt.plot(time_values, snr_metric_values, label="SNR")
plt.xlabel("Sample")
plt.ylabel("Normalized Metric Value")
plt.title("Evaluation: Metrics Over Test Set")
plt.legend()
plt.savefig(Path(args.results_dir) / 'eval_metrics_plot.png')

o_plot = output.detach().cpu().numpy().reshape(-1)
t_plot = target.detach().cpu().numpy().reshape(-1)
i_plot = input.detach().cpu().numpy().reshape(-1)

plot_compare_waveform(t_plot, o_plot)
plot_zoom_waveform(t_plot, o_plot,args.sr, t_start=0.5, t_end=0.6)
plot_compare_spectrogram(t_float, o_float, i_float,  titles=['target', 'output', 'input'], ylabel="freq_bin", aspect="auto", xmax=None)

