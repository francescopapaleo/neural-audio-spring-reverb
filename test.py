from pathlib import Path

import torch
import torchaudio
import torchsummary
import torchvision
import auraloss
import pickle
from matplotlib import pyplot as plt
import numpy as np

from data import SpringDataset
from sandbox.utcn import TCN, causal_crop
from utils.plot import plot_compare_waveform, plot_zoom_waveform, plot_compare_spectrogram, plot_metrics, save_plot


def main():
    
    sample_rate = args.sr

    print("## Loading data...")
    dataset = SpringDataset(root_dir=args.data_dir, split='test')
    testloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    
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

    torchsummary.summary(model, [(1,65536), (1,2)], device=args.device)

    print("## Initializing metrics...")

    # Initialize lists for storing metric values
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    stft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048],
        win_lengths=[32, 128, 512, 2048],
        hop_sizes=[16, 64, 256, 1024])
    esr = auraloss.time.ESRLoss()
    dc = auraloss.time.DCLoss()
    snr = auraloss.time.SNRLoss()

    test_results = {
        "mse": [],
        "l1": [],
        "stft": [],
        "esr": [],
        "dc": [],
        "snr": []
    }

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    print("## Testing...")
    model.eval()
    with torch.no_grad():
        for input, target in testloader:

            input = input.float().to(args.device)
            target = target.float().to(args.device)
            c = torch.tensor([0.0, 0.0], device=device).view(1,1,-1)

            rf = model.compute_receptive_field()
            input_pad = torch.nn.functional.pad(input, (rf-1, 0))

            output = model(input_pad)

            # Calculate the metrics
            test_results['mse'].append(mse(output, target))
            test_results['l1'].append(l1(output, target))
            test_results['stft'].append(stft(output, target))
            test_results['esr'].append(esr(output, target))
            test_results['dc'].append(dc(output, target))
            test_results['snr'].append(snr(output, target))

    # Save metric data
    with open(Path(args.results_dir) / 'eval_metrics.pkl', 'wb') as f:
        pickle.dump(test_results, f)
    
    # Normalize
    for name, values in test_results.items():
        values = (values - np.mean(values)) / np.std(values)

    print('Saving audio files...')
    ofile = Path(args.results_dir) / 'eval_output.wav'
    o_float = output.view(1,-1).cpu().float()
    torchaudio.save(ofile, o_float, sample_rate)
    
    tfile = Path(args.results_dir) / 'eval_target.wav'
    t_float = target.view(1,-1).cpu().float()
    torchaudio.save(tfile, t_float, sample_rate)

    ifile = Path(args.results_dir) / 'eval_input.wav'
    i_float = input.view(1,-1).cpu().float()
    torchaudio.save(ifile, i_float, sample_rate)
    
    print('Plotting...')

    plot_metrics(test_results, args)

    o_plot = output.detach().cpu().numpy().reshape(-1)
    t_plot = target.detach().cpu().numpy().reshape(-1)
    i_plot = input.detach().cpu().numpy().reshape(-1)

    plot_compare_waveform(t_plot, o_plot)
    plot_zoom_waveform(t_plot, o_plot, args.sr, t_start=0.5, t_end=0.6, results_dir=args.results_dir)
    plot_compare_spectrogram(t_float, o_float, i_float,  titles=['target', 'output', 'input'], ylabel="freq_bin", aspect="auto", xmax=None)

if __name__ == "__main__":
    main()