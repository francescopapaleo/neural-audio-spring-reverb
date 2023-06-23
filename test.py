#test.py

import torch
import auraloss
from datetime import datetime
from data.dataset import SpringDataset
from argparse import ArgumentParser
from pathlib import Path
from utils.plotter import plot_compare_waveform, plot_compare_spectrogram
from utils.helpers import load_data, initialize_model, save_model_checkpoint

torch.manual_seed(42)            
def testing(load, datadir, logdir, audiodir, device, sample_rate):

    # Parse command line arguments
    args = parse_args()

    # Define hyperparameters
    hparams = {
        'n_inputs': 1,
        'n_outputs': 1,
        'n_blocks': 10,
        'kernel_size': 15,
        'n_channels': 64,
        'dilation_growth': 2,
        'cond_dim': 0,
    }

    # Define loss function and optimizer
    device = torch.device(args.device)
    criterion = auraloss.freq.STFTLoss().to(device)  
    esr = auraloss.time.ESRLoss().to(device)

    # Initialize model
    model = initialize_model(device, "TCN", hparams)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ms1 = int(args.n_epochs * 0.8)
    ms2 = int(args.n_epochs * 0.95)
    milestones = [ms1, ms2]
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        gamma=0.1,
        verbose=False,
    )

    # Initialize Tensorboard writer
    log_dir = Path(args.logdir) / f"tcn_{args.n_epochs}_{args.batch_size}_{args.lr}"
    writer = SummaryWriter(log_dir=log_dir)

    # Load data
    _, _, test_loader = load_data(args.datadir, args.batch_size)

    # initialize tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_run = f"{model_name}_{n_epochs}_{batch_size}_{lr}_{timestamp}"
    writer = SummaryWriter(log_dir=f'{logdir}/{current_run}')

    print("## Initializing metrics...")
    l1 = torch.nn.L1Loss()
    esr = auraloss.time.ESRLoss()
    dc = auraloss.time.DCLoss()
    snr = auraloss.time.SNRLoss()
    
    criterions = [l1, esr, dc, snr]
    test_results = {"l1": [], "esr": [], "dc": [], "snr": []}

    # dummy condition tensor    
    c = torch.tensor([0.0, 0.0]).view(1,1,-1)           
    
    print("## Testing...")

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
            rf = model.compute_receptive_field()
            input_pad = torch.nn.functional.pad(input, (rf-1, 0))
            target_pad = torch.nn.functional.pad(target, (rf-1, 0))
            
            # forward pass
            output = model(input_pad, c)
            output_trim = output[:,:,:target_pad.size(2)]

            # Compute metrics means for current batch
            for metric, name in zip(criterions, test_results.keys()):
                batch_score = metric(output_trim, target_pad).item()
                test_results[name].append(batch_score)
                
                # Write metrics to tensorboard
                writer.add_scalar(f'test/batch_{name}', batch_score, global_step)

            # Plot and save audios every 4 batches
            if step % 4 ==0:
                single_target = target_pad[0]
                single_output = output_trim[0]

                waveform_fig = plot_compare_waveform(single_target.detach().cpu(), 
                                                    single_output.detach().cpu(),
                                                    sample_rate,
                                                    title=f"Waveform_{model_name}_{global_step}"
                                                    )
                spectrogram_fig = plot_compare_spectrogram(single_target.detach().cpu(), 
                                                        single_output.detach().cpu(), 
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

    print("## Computing global metrics...")
    # compute global metrics means
    for name in test_results.keys():
        global_score = sum(test_results[name]) / len(test_results[name])
        writer.add_scalar(f'test/global_{name}', global_score, global_step)

    print("## Saving results...")
    # Flush and close the writer
    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, default='../../datasets/plate-spring/spring/', help='Path (rel) to dataset ')
    parser.add_argument('--audiodir', type=str, default='../audio/processed/', help='Path (rel) to audio files')
    parser.add_argument('--logdir', type=str, default='../results/runs', help='name of the log directory')
    parser.add_argument('--load', type=str, required=True, help='Path (rel) to checkpoint to load')
    parser.add_argument('--device', type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", help='set device to run the model on')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate of the audio')
    
    args = parser.parse_args()

    testing(args.load, args.datadir, args.logdir, args.audiodir, args.device, args.sample_rate)
    