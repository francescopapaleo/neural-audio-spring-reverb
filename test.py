#test.py

import torch
import torchaudio
import torchaudio.functional as F
import auraloss
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.plotter import plot_compare_waveform, plot_compare_spectrogram
from src.helpers import load_data, select_device, load_model_checkpoint
from configurations import parse_args

torch.manual_seed(42)
torch.backends.cudnn.enabled = True

# def soft_clip(tensor, limit=0.95):
#     return limit * torch.tanh(tensor / limit)

def main():
    print("Testing model...")
    args = parse_args()
    torch.cuda.empty_cache()
    
    device = select_device(args.device)
    sample_rate = args.sample_rate

    model, model_name, hparams = load_model_checkpoint(device, args.checkpoint, args)
    
    batch_size = hparams['batch_size']
    n_epochs = hparams['n_epochs']
    lr = hparams['lr']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.logdir) / f"test/{model_name}_{n_epochs}_{batch_size}_{lr}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    _, _, test_loader = load_data(args.datadir, batch_size)

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
        for step, (dry, wet) in enumerate(test_loader):
            input = dry
            target = wet            
            global_step = step + 1
            print(f"Batch {global_step}/{num_batches}")

            # move input and target to device
            input = input.to(device)                    
            target = target.to(device)
            c = c.to(device)
            
            # forward pass
            output = model(input)
            # output = soft_clip(output)
            output = F.highpass_biquad(output, sample_rate, 20)
            output /= torch.max(torch.abs(output))

            # Compute metrics means for current batch
            for metric, name in zip(criterions, test_results.keys()):
                batch_score = metric(output, target).item()
                test_results[name].append(batch_score)

            input = input.squeeze(1).detach().cpu()
            target = target.squeeze(1).detach().cpu()
            output = output.squeeze(1).detach().cpu()

            # Plot and save audios every n batches
            if step % batch_size == 0:
                
                torchaudio.save(f"Input_{model_name}_{global_step}.wav", input, sample_rate, bits_per_sample=24)
                torchaudio.save(f"Target_{model_name}_{global_step}.wav", target, sample_rate, bits_per_sample=24)
                torchaudio.save(f"Output_{model_name}_{global_step}.wav", output, sample_rate, bits_per_sample=24)

                waveform_fig = plot_compare_waveform(
                    target,output, sample_rate,title=f"Waveform_{model_name}_{global_step}")
                
                spectrogram_fig = plot_compare_spectrogram(
                    target,output,sample_rate,title=f"Spectra_{model_name}_{global_step}",
                    t_label=f"Target_{global_step}", o_label=f"Output_{global_step}")

                writer.add_figure(f"test/Waveform_{model_name}_{global_step}", waveform_fig, global_step)
                writer.add_figure(f"test/Spectra_{model_name}_{global_step}", spectrogram_fig, global_step)
                
                # writer.add_audio(f"test/Input_{model_name}_{global_step}", 
                #                 single_input.detach().cpu(), global_step, sample_rate=sample_rate)
                # writer.add_audio(f"test/Target_{model_name}_{global_step}", 
                #                 single_target.detach().cpu(), global_step, sample_rate=sample_rate)
                # writer.add_audio(f"test/Output_{model_name}_{global_step}", 
                #                 single_output.detach().cpu(), global_step, sample_rate=sample_rate)

    for name in test_results.keys():
        global_score = sum(test_results[name]) / len(test_results[name])
        writer.add_scalar(f'test/global_{name}', global_score, global_step)
    
    mean_test_results = {k: sum(v) / len(v) for k, v in test_results.items()}
    writer.add_hparams(hparams, mean_test_results)

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
