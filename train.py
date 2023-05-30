from tcn import TCN
from data import SpringDataset
from pathlib import Path
from datetime import datetime

import argparse
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import auraloss
import numpy as np
from tqdm import trange, tqdm
import time

from torch.utils.tensorboard import SummaryWriter
from torch.monitor import TensorboardEventHandler, register_event_handler

torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.empty_cache()
    
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./data/plate-spring/spring/', help='dataset')
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--crop', type=int, default=3200)

args = parser.parse_args()

def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs = [torch.from_numpy(x).to(args.device) for x in inputs]
    targets = [torch.from_numpy(x).to(args.device) for x in targets]
    return torch.stack(inputs), torch.stack(targets)

def training():
    print("#-----------------------------------------------------------------------#")
    print("                     Initializing training process")  
    print("-------------------------------------------------------------------------")
    print("")
    
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    sample_rate = 16000
    global crop_len 
    crop_len = args.crop

    if args.device is None: 
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Torch version: {torch.__version__} ------ Selected Device: {args.device}')
    print(f'Sample Rate: {sample_rate} Hz ------  Crop Lenght: {crop_len} samples')
    print("-------------------------------------------------------------------------")
    
    dataset = SpringDataset(root_dir=args.data_dir, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=0, shuffle=True, drop_last=True, collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=0, shuffle=False, drop_last=True, collate_fn=collate_fn)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                            # timestamp for tensorboard
    writer = SummaryWriter(f'runs/tcn_{n_epochs}_{batch_size}_{timestamp}')            # tensorboard writer
    register_event_handler(TensorboardEventHandler(writer))                         # register event handler

    # Hyperparameters
    hparams = {
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'lr': 0.01,
        'sched_gamma': 0.1,
        'n_inputs': 1,
        'n_outputs': 1,
        'n_blocks': 7,
        'kernel_size': 11,
        'n_channels': 64,
        'dilation_growth': 4,
        'cond_dim': 0,
    }

    # define the model
    model = TCN(
        n_inputs = hparams['n_inputs'], 
        n_outputs = hparams['n_outputs'], 
        n_blocks = hparams['n_blocks'],
        kernel_size = hparams['kernel_size'],
        n_channels = hparams['n_channels'], 
        dilation_growth = hparams['dilation_growth'],
        cond_dim = hparams['cond_dim'],
    )
    model.to(args.device)
    
    writer.add_hparams(hparams, {})

    rf = model.compute_receptive_field()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-------------------------------------------------------------------------")
    print(f"Parameters: {params*1e-3:0.3f} k")
    print(f"Receptive field: {rf} samples or {(rf / sample_rate)*1e3:0.1f} ms")       
    print("-------------------------------------------------------------------------")
    
    # define the loss function
    criterion = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048], 
        win_lengths=[32, 128, 512, 2048],
        hop_sizes=[16, 64, 256, 1024]).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # optimizer
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1, verbose=False
        )
    
    c = torch.tensor([0.0, 0.0]).view(1,1,-1)

    ############################### Training Loop #################################
    
    min_valid_loss = np.inf
    
    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        
        for batch_idx, (input, target) in enumerate(train_loader): 
            optimizer.zero_grad()

            input = input.to(args.device)                   # move input and target to device
            target = target.to(args.device)
            c = c.to(args.device)
            
            start_idx = rf                                  # crop input and target
            stop_idx = start_idx + crop_len
            if stop_idx > input.shape[-1]:
                stop_idx = input.shape[-1]
                start_idx = stop_idx - crop_len
            input_crop = input[:, :, start_idx:stop_idx]
            target_crop = target[:, :, start_idx:stop_idx]     

            output = model(input_crop, c)                   # forward pass

            loss = criterion(output, target_crop)              # compute loss
            
            loss.backward()                                 # compute gradients
            optimizer.step()                                # update weights

            train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            
            writer.add_scalar('Batch Loss', loss.item(), global_step = batch_idx + 1 + epoch * len(train_loader))
        
        writer.add_scalar('Training Loss', avg_train_loss, global_step = epoch * len(train_loader))
        writer.flush()

        #################################### Validation Loop #########################################
        model.eval()
        with torch.no_grad():
            
            for step, (input, target) in enumerate(valid_loader):
                input = input.to(args.device)               # move input and target to device
                target = target.to(args.device)
                c = c.to(args.device)

                start_idx = rf                              # crop input and target
                stop_idx = start_idx + crop_len
                if stop_idx > input.shape[-1]:
                    stop_idx = input.shape[-1]
                    start_idx = stop_idx - crop_len
                input_crop = input[:, :, start_idx:stop_idx]
                target_crop = target[:, :, start_idx:stop_idx]     

                output = model(input_crop, c)               # forward pass

                loss = criterion(output, target_crop)          # compute loss
                
                valid_loss += loss.item()                   # accumulate loss
                avg_valid_loss = valid_loss / len(valid_loader)

                writer.add_scalar('Validation loss', avg_valid_loss, global_step = epoch * len(valid_loader))
                writer.flush()

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) Saving model ...')
                save_to = f'runs/tcn_{n_epochs}_{batch_size}_{timestamp}/tcn_ckpt_{n_epochs}.pth'
                torch.save(model.state_dict(), save_to)
               
                min_valid_loss = valid_loss    
            
        print(f'Epoch {epoch +1} \t\t Validation Loss: {avg_valid_loss:.6f}, \t\t Training Loss: {avg_train_loss:.6f}', end='\r')
        scheduler.step()
    
    writer.add_graph(model, input_to_model=input, verbose=False)
    writer.flush()
    writer.close()        

    print('                         Training Completed!')
    print("#-----------------------------------------------------------------------#")
    
if __name__ == "__main__":    
    training()

