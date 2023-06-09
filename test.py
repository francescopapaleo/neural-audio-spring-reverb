""" Testing script
Test a trained model on the test set and save the results with Tensorboard.
"""

import torch
import torchaudio
import auraloss
from datetime import datetime
from data import SpringDataset
from argparse import ArgumentParser
from pathlib import Path

from TCN import TCNBase
from utils.plotter import plot_compare_waveform, plot_zoom_waveform, plot_compare_spectrogram

def testing(load, data_dir, device, sample_rate):

    # set device                                                                                
    if device is None:                                                              
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load from checkpoint                                                                                                        
    try:
        checkpoint = torch.load(load, map_location=device)
        hparams = checkpoint['hparams']
    except Exception as e:
        print(f"Failed to load model state: {e}")
        return
    
    # instantiate model 
    model = TCNBase(                                                        
        n_inputs = hparams['n_inputs'], 
        n_outputs = hparams['n_outputs'], 
        n_blocks = hparams['n_blocks'],
        kernel_size = hparams['kernel_size'],
        n_channels = hparams['n_channels'], 
        dilation_growth = hparams['dilation_growth'],
        cond_dim = hparams['cond_dim'],
    ).to(device)
    
    # batch_size = hparams['batch_size']
    batch_size = 16
    n_epochs = hparams['n_epochs']
    lr = hparams['l_rate']

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Failed to load model state: {e}")
        return
    
    # load test dataset
    dataset = SpringDataset(root_dir=data_dir, split='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=0, drop_last=True)

    # initialize tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'runs/01B_test/tcn_{n_epochs}_{batch_size}_{lr}_{timestamp}')
    

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
    middle_batch_idx = num_batches // 2
    
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(test_loader):
            global_step = step + 1

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

             # Save and plot for the last batch only
            if step == num_batches - 1:
                single_target = target_pad[0]
                single_output = output_trim[0]

                # plot waveforms
                plot_compare_waveform(single_target.cpu(), single_output.cpu(), f"waveform_comparison_last_batch")


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

    parser.add_argument('--load', type=str, default= 'checkpoints/tcn_50_16_0.001_20230605_190459.pt', help='checkpoint rel path to load')
    parser.add_argument('--data_dir', type=str, default='../plate-spring/spring/', help='dataset rel path')
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--sample_rate', type=int, default=16000)
    
    args = parser.parse_args()

    # for file in Path('./checkpoints').glob('tcn_*'):
    #     args.load = file
    testing(args.load, args.data_dir, args.device, args.sample_rate)
    