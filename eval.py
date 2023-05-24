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
from utils.plot import plot_compare_waveform, plot_zoom_waveform

import pyloudnorm as pyln
from scipy.io import wavfile

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

print("## Evaluation...")

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

plt.figure(figsize=(12, 6))
plt.plot(time_values, l1_loss_values, label="L1 Loss")
plt.plot(time_values, stft_loss_values, label="STFT Loss")
plt.plot(time_values, lufs_diff_values, label="LUFS Difference")
plt.plot(time_values, aggregate_loss_values, label="Aggregate Loss")
plt.xlabel("Time")
plt.ylabel("Metric Value")
plt.title("Metrics Over Time")
plt.legend()
plt.savefig(Path(args.results_dir) / 'eval_metrics_plot.png')

# def plot_zoom_waveform(y, y_pred, sample_rate, t_start=None, t_end=None):
    # '''Plot the waveform of the ground truth and the prediction
    # Parameters
    # ----------
    # y : array_like
    #     Ground truth signal
    # y_pred : array_like
    #     The predicted signal
    # fs : int, optional
    #     The sampling frequency (default to 1, i.e., samples).
    # t_start : float, optional
    #     The start time of the plot (default to None).
    # t_end : float, optional
    #     The end time of the plot (default to None).'''
    
    # # Create a time array
    # t = np.arange(y.shape[0]) / sample_rate

    # Determine the indices corresponding to the start and end times
    # if t_start is not None:
    #     i_start = int(t_start * sample_rate)
    # else:
    #     i_start = 0

    # if t_end is not None:
    #     i_end = int(t_end * sample_rate)
    # else:
    #     i_end = len(t)

    # fig, ax = plt.subplots(nrows=1, ncols=1)

    # ax.plot(t[i_start:i_end], y[i_start:i_end], alpha=0.7, label='Ground Truth', color='blue')
    # ax.plot(t[i_start:i_end], y_pred[i_start:i_end], alpha=0.7, label='Prediction', color='red')

    # ax.set_title('Waveform')
    # ax.set_xlabel('Time [s]')
    # ax.set_ylabel('Amplitude')
    # ax.grid(True)
    # ax.legend()

    # plt.savefig(Path(args.results_dir) / 'waveform_zoom.png')
    # plt.close(fig)
    # print("Saved zoomed waveform plot to: ", Path(args.results_dir) / 'waveform_zoom.png')

t_plot = target.detach().cpu().numpy().reshape(-1)
o_plot = output.detach().cpu().numpy().reshape(-1)

plot_compare_waveform(t_plot, o_plot)
plot_zoom_waveform(t_plot, o_plot,args.sr, t_start=0.5, t_end=0.6)