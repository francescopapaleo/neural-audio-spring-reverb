# test.py

import torch
import auraloss
import numpy as np
from datetime import datetime
from tcn import TCN
from data import SpringDataset
from argparse import ArgumentParser
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

def testing(args):
    data_dir = args.data_dir
    device = args.device
    sample_rate = args.sample_rate
    lr = args.lr
    
    if device is None: 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    load_from = Path('./checkpoints') / args.load
    try:
        checkpoint = torch.load(load_from, map_location=device)
        hparams = checkpoint['hparams']
    except Exception as e:
        print(f"Failed to load model state: {e}")
        return
    
    model = TCN(                                                    # instantiate model     
        n_inputs = hparams['n_inputs'], 
        n_outputs = hparams['n_outputs'], 
        n_blocks = hparams['n_blocks'],
        kernel_size = hparams['kernel_size'],
        n_channels = hparams['n_channels'], 
        dilation_growth = hparams['dilation_growth'],
        cond_dim = hparams['cond_dim'],
    ).to(device)
    
    batch_size = hparams['batch_size']
    n_epochs = hparams['n_epochs']

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Failed to load model state: {e}")
        return
    
    dataset = SpringDataset(root_dir=data_dir, split='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=0, drop_last=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'results/tcn_{n_epochs}_{batch_size}_{lr}_{timestamp}')
    
    print("## Initializing metrics...")
    l1 = torch.nn.L1Loss()
    esr = auraloss.time.ESRLoss()
    dc = auraloss.time.DCLoss()
    snr = auraloss.time.SNRLoss()

    test_results = {"l1": [], "esr": [], "dc": [], "snr": []}

    c = torch.tensor([0.0, 0.0]).view(1,1,-1)           # dummy condition tensor
    model = model.to(device)                           # move model to device
    
    print("## Testing...")
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(test_loader):
            global_step = step + 1

            input = input.to(device)                    # move input and target to device
            target = target.to(device)
            c = c.to(device)

            rf = model.compute_receptive_field()
            input_pad = torch.nn.functional.pad(input, (rf-1, 0))
            target_pad = torch.nn.functional.pad(target, (rf-1, 0))
            
            output = model(input_pad, c)
            output_trimmed = output[:,:,:target_pad.size(2)]

            # Calculate the metrics AND COMPUTE MEANS, RECORD JUST ONE SCORE PER METRIC 
            test_results['l1'].append(l1(output_trimmed, target))
            test_results['esr'].append(esr(output_trimmed, target))
            test_results['dc'].append(dc(output_trimmed, target))
            test_results['snr'].append(snr(output_trimmed, target))
    

    # Log metrics to TensorBoard
    writer.add_scalar('test/l1', test_results['l1'][-1].item(), global_step)
    writer.add_scalar('test/esr', test_results['esr'][-1].item(), global_step)
    writer.add_scalar('test/dc', test_results['dc'][-1].item(), global_step)
    writer.add_scalar('test/snr', test_results['snr'][-1].item(), global_step)

    print("## Saving results...")
    # Flush and close the writer
    writer.flush()
    writer.close()
  
if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='../plate-spring/spring/', help='dataset')
    parser.add_argument('--n_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--crop', type=int, default=3200)
    parser.add_argument('--sample_rate', type=int, default=16000)

    parser.add_argument('--load', type=str, default=None, help='checkpoint to load')

    args = parser.parse_args()

    testing(args)

    