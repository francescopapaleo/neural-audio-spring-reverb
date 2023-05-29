# python train.py --batch_size 4 --epochs 1 --device cuda:0
import argparse
from tcn import TCN
from utils.data import SpringDataset
from pathlib import Path
from datetime import datetime
import logging
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.monitor import TensorboardEventHandler, register_event_handler
import torch.nn.functional as F
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torchaudio.transforms as T
import auraloss

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/plate-spring/spring/', help='dataset')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='state dict')
parser.add_argument('--batch_size', type= int, default=2)
parser.add_argument('--epochs', type= int, default=10)
parser.add_argument('--device', type=str)
parser.add_argument('--lenght', type=int, default=3200)

def main():
    print("#-----------------------------------------------------------------------#")
    print("                     Initializing training process")  
    print("-------------------------------------------------------------------------")
    print("")
    
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    sr = 16000
    global crop_len 
    crop_len = args.lenght

    if args.device is None: 
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Torch version: {torch.__version__} ------ Selected Device: {args.device}')
    print(f'Sample Rate: {sr} Hz ------  Crop Lenght: {crop_len} samples')
    print("-------------------------------------------------------------------------")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    writer = SummaryWriter('logs/tcn_{}'.format(timestamp))
    register_event_handler(TensorboardEventHandler(writer))
    
    dataset = SpringDataset(root_dir=args.data_dir, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=4, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=4, shuffle=False)

    model = TCN(
        n_inputs=1, 
        n_outputs=1, 
        n_blocks=7,
        kernel_size=11,
        n_channels=64,
        dilation_growth=4,
        cond_dim=0,
    )
    model.to(args.device)
    print(f"Model: {model._get_name()}")

    rf = model.compute_receptive_field()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params*1e-3:0.3f} k")
    print(f"Receptive field: {rf} samples or {(rf / sr)*1e3:0.1f} ms")       
    print("-------------------------------------------------------------------------")

    stft = auraloss.freq.STFTLoss().to(args.device)
    mse = auraloss.time.MSELoss().to(args.device)
    criterion = stft

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # optimizer
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1, verbose=True)
    
    c = torch.tensor([0.0, 0.0]).view(1,1,-1)

    ############################### Training Loop #################################
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('logs/tcn_{}'.format(timestamp)),
        record_shapes=False,
        profile_memory=True,
        with_stack=True
        ) as prof:
        running_loss = 0.0              # initialize running_loss
        min_valid_loss = np.inf             # initialize min_valid_loss
        
        model.train()
        for e in (range(epochs)):      # iterate over epochs
            print(f"Epoch: {e+1} / {epochs}", end='\r')
            train_loss = 0.0           # initialize train_loss 

            p_bar = tqdm(enumerate(train_loader), total=len(train_loader),leave=False,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
                        dynamic_ncols=False)
            for batch_idx, (input, target) in p_bar:
                p_bar.set_description(f"Epoch {e+1} / {epochs} | Step {batch_idx+1} / {batch_idx}")
                optimizer.zero_grad()               # clear gradients

                input = input.to(args.device)       # move everything to device
                target = target.to(args.device)
                c = c.to(args.device)
                # print(f"input shape: {input.shape}, target shape: {target.shape}", end='\r')
                
                start_idx = rf
                stop_idx = start_idx + crop_len
                # print(f"start_idx: {start_idx}, stop_idx: {stop_idx}, receptive field: {rf}", end='\r')

                if stop_idx > input.shape[-1]:
                    stop_idx = input.shape[-1]
                    start_idx = stop_idx - crop_len

                input_crop = input[:, :, start_idx:stop_idx]
                target_crop = target[:, :, start_idx:stop_idx]     

                output = model(input_crop, c)  # forward pass
                loss = criterion(output, target_crop)  # compute loss
                loss.backward()  # compute gradients
                optimizer.step()  # update weights

                train_loss += loss.item()
                running_loss += loss.item()
                writer.add_scalar('train_loss', train_loss, global_step=e)
                writer.add_scalar('running_loss', running_loss, global_step=e)
                writer.close()    
                prof.step()
                try:
                    prof.export_chrome_trace(f"logs/prof.pt.trace{timestamp}.json")
                except Exception as err:
                    print("Could not write trace file: ", err)

            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)}')
            
            #################################### Validation Loop #########################################
            valid_loss = 0.0                                               # initialize valid_loss
            model.eval()                  
            for step, (input, target) in enumerate(valid_loader):          # iterate over batches
                print(f"Validation step:{step}", end='\r')

                input = input.to(args.device)
                target = target.to(args.device)
                c = c.to(args.device)

                output = model(input, c)
                loss = criterion(output, target)
                valid_loss += loss.item() * input.size(0)

            valid_loss /= len(valid_loader)
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss}')
            writer.add_scalar('valid_loss', valid_loss, global_step=e)

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving model ...')
                min_valid_loss = valid_loss

                save_to = 'checkpoints/tcn{}.pth'.format(timestamp)
                torch.save(model.state_dict(), save_to)         

            scheduler.step()            # update learning rate
            writer.add_graph(model, input_to_model=input, verbose=False)
            writer.flush()
        
    
    print('                         Finished Training')
    print("#-----------------------------------------------------------------------#")
    

if __name__ == "__main__":    
    main()
