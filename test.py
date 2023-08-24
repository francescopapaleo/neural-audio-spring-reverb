import torch
import torchaudio
import torchaudio.functional as F
import auraloss
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from src.plotter import plot_compare_waveform, plot_compare_spectrogram
from src.helpers import load_data, select_device, load_model_checkpoint
from configurations import parse_args

def main():
    print("Testing model...")
    args = parse_args()
    torch.manual_seed(42)
    
    device = select_device(args.device)

    torch.backends.cudnn.deterministic = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    torch.cuda.empty_cache()

    sample_rate = args.sample_rate

    model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params = load_model_checkpoint(device, args.checkpoint, args)
    
    batch_size = hparams['batch_size']
    n_epochs = hparams['n_epochs']
    lr = hparams['lr']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.logdir) / f"test/{model_name}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    _, _, test_loader = load_data(args.datadir, batch_size)

    mae = torch.nn.L1Loss()
    mse = torch.nn.MSELoss()
    esr = auraloss.time.ESRLoss()
    dc = auraloss.time.DCLoss()
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()

    criterions = [mae, mse, esr, dc, mrstft]
    test_results = {"mae": [], "esr": [], "dc": [], "mrstft": []}

    num_batches = len(test_loader)
    rtf_list = []

    model.eval()
    with torch.no_grad():
        for step, (dry, wet) in enumerate(test_loader):
            start_time = datetime.now()
            
            input = dry
            target = wet            
            global_step = step + 1
            print(f"Batch {global_step}/{num_batches}")

            # move input and target to device
            input = input.to(device)                    
            target = target.to(device)

            # forward pass
            output = model(input)
            
            end_time = datetime.now()
            duration = end_time - start_time
            num_samples = input.size(2)
            lenght_in_seconds = num_samples / sample_rate
            rtf = duration.total_seconds() / lenght_in_seconds
            rtf_list.append(rtf)

            # Compute metrics means for current batch
            for metric, name in zip(criterions, test_results.keys()):
                batch_score = metric(output, target).item()
                test_results[name].append(batch_score)
            
            # Plot and save audios every n batches
            if step == num_batches - 1:

                inp = input.view(-1).unsqueeze(0).cpu()
                tgt = target.view(-1).unsqueeze(0).cpu()
                out = output.view(-1).unsqueeze(0).cpu()

                inp /= torch.max(torch.abs(inp))
                tgt /= torch.max(torch.abs(tgt))                
                out /= torch.max(torch.abs(out))

                save_in = f"{log_dir}/inp_{hparams['conf_name']}.wav"
                torchaudio.save(save_in, inp, args.sample_rate, bits_per_sample=24)

                save_out = f"{log_dir}/out_{hparams['conf_name']}.wav"
                torchaudio.save(save_out, out, args.sample_rate, bits_per_sample=24)

                save_target = f"{log_dir}/tgt_{hparams['conf_name']}.wav"
                torchaudio.save(save_target, tgt, args.sample_rate, bits_per_sample=24)

    for name in test_results.keys():
        global_score = sum(test_results[name]) / len(test_results[name])
        writer.add_scalar(f'test/global_{name}', global_score, global_step)
    
    mean_test_results = {k: sum(v) / len(v) for k, v in test_results.items()}
    avg_rtf = sum(rtf_list) / len(rtf_list)
    mean_test_results['rtf'] = avg_rtf

    writer.add_hparams(hparams, mean_test_results)

    # writer.add_scalar(f'test/rtf', avg_rtf, global_step)

    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
