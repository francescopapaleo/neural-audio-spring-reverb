#test.py

import torch
import auraloss
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.plotter import plot_compare_waveform, plot_compare_spectrogram
from src.helpers import load_data, select_device, load_model_checkpoint
from configurations import parse_args

torch.manual_seed(42)

def evaluate_model(model, device, model_name, hparams, test_loader, writer, sample_rate):
    mae = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    esr = auraloss.time.ESRLoss()
    stft = auraloss.freq.STFTLoss()
    dc = auraloss.time.DCLoss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    criterions = [mae, mse, esr, dc, mrstft]
    test_results = {"mae": [], "mse": [], "esr": [], "dc": [], "mrstft": []}

    c = torch.tensor([0.0, 0.0]).view(1,1,-1)           

    num_batches = len(test_loader)

    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(test_loader):
            global_step = step + 1
            print(f"Batch {global_step}/{num_batches}")

            # move input and target to device
            input = input.to(device)                    
            target = target.to(device)
            c = c.to(device)

            # pad input and target to match receptive field
            # Conditionally compute the receptive field for certain model types
            if hparams['model_type'] in ["TCN", "WaveNet"]:
                rf = model.compute_receptive_field()

                input_pad = torch.nn.functional.pad(input, (rf-1, 0))
                target_pad = torch.nn.functional.pad(target, (rf-1, 0))
            else:
                rf = None
                input_pad = input
                target_pad = target

            # forward pass
            output = model(input_pad, c)
            output_trim = output[:,:,:target_pad.size(2)]

            # Compute metrics means for current batch
            for metric, name in zip(criterions, test_results.keys()):
                batch_score = metric(output_trim, target_pad).item()
                test_results[name].append(batch_score)

            # Plot and save audios every 4 batches
            if step % 8 ==0:
                single_target = target_pad[0]
                single_output = output_trim[0]

                waveform_fig = plot_compare_waveform(single_target.detach().cpu(), 
                                                    single_output.detach().cpu(),
                                                    sample_rate,
                                                    title=f"Waveform_{model_name}_{global_step}"
                                                    )
                spectrogram_fig = plot_compare_spectrogram(single_target.cpu(), 
                                                        single_output.cpu(), 
                                                        sample_rate,
                                                        title=f"Spectra_{model_name}_{global_step}",
                                                        t_label=f"Target_{global_step}", o_label=f"Output_{global_step}"
                                                        )

                writer.add_figure(f"test/Waveform_{model_name}_{global_step}", waveform_fig, global_step)
                writer.add_figure(f"test/Spectra_{model_name}_{global_step}", spectrogram_fig, global_step)

                writer.add_audio(f"test/Target_{model_name}_{global_step}", 
                                single_target.detach().cpu(), global_step, sample_rate=sample_rate)
                writer.add_audio(f"test/Output_{model_name}_{global_step}", 
                                single_output.detach().cpu(), global_step, sample_rate=sample_rate)

    return test_results, global_step
    

def main():
    print("Testing model...")
    args = parse_args()
    torch.cuda.empty_cache()
    
    device = select_device(args.device)

    model, model_name, hparams = load_model_checkpoint(device, args.checkpoint_path)
    
    batch_size = hparams['batch_size']
    n_epochs = hparams['n_epochs']
    lr = hparams['lr']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.logdir) / f"test/{model_name}_{n_epochs}_{batch_size}_{lr}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)

    _, _, test_loader = load_data(args.datadir, batch_size)

    test_results, global_step = evaluate_model(model, device, model_name, hparams, test_loader, writer, args.sample_rate)

    for name in test_results.keys():
        global_score = sum(test_results[name]) / len(test_results[name])
        writer.add_scalar(f'test/global_{name}', global_score, global_step)
    
    mean_test_results = {k: sum(v) / len(v) for k, v in test_results.items()}
    writer.add_hparams(hparams, mean_test_results)

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
    

