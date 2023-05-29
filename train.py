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
import torch.autograd.profiler as profiler
import os
import torchaudio.transforms as T
import auraloss

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.empty_cache()

l = logging.INFO
logging.basicConfig(level=l, format="%(levelname)s : %(message)s")
info = logging.info

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/plate-spring/spring/', help='dataset')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='state dict')
parser.add_argument('--batch_size', type= int, default=2)
parser.add_argument('--epochs', type= int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')

def main():
################################## Training Loop ###########################################
    print("#-----------------------------------------------------------------------#")
    print("                     Initializing training process")  
    print("-------------------------------------------------------------------------")
    print("")
    
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size

    print(f'Torch version: {torch.__version__}'), print(f'Cuda available: {torch.cuda.is_available()}')

    if torch.cuda.is_available() is False:
        torch.set_default_dtype(torch.float32)  
        torch.set_default_tensor_type(torch.FloatTensor) 
        args.device = 'cpu'
        print("Using CPU")
    else:
        pass

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    writer = SummaryWriter('logs/tcn_{}'.format(timestamp))
    register_event_handler(TensorboardEventHandler(writer))

    dataset = SpringDataset(root_dir=args.data_dir, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=0, shuffle=False)

    lenght = 3200

    model = TCN(
        n_inputs=1, 
        n_outputs=1, 
        n_blocks=1,
        kernel_size=13,
        n_channels=64,
        dilation_growth=4,
        cond_dim=0,
    )
    model.to(args.device)
    rf = model.compute_receptive_field()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
           
    criterion = torch.nn.MSELoss().to(args.device)   

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # optimizer
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1, verbose=True)

    # Profiling
    with profiler.profile(use_cuda=True, use_kineto=True, record_shapes=True) as prof:
        global_step = 0                     # initialize global_step
        train_loss = 0.0                    # initialize train_loss 
        valid_loss = 0.0                    # initialize valid_loss
        min_valid_loss = np.inf             # initialize min_valid_loss
        
        ############################### Training Loop #################################
        model.train()

        for e in (range(epochs)):      # iterate over epochs
            
            for batch_idx, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
                # print(f"step:{step}", end='\r')
                optimizer.zero_grad()               # clear gradients

                c = torch.tensor([0.0, 0.0]).view(1,1,-1)

                input = input.to(args.device)           # move everything to device
                target = target.to(args.device)
                c = c.to(args.device)
                
                input_pad = torch.nn.functional.pad(input, (rf-1, 0))
                
                start_idx = rf
                stop_idx = start_idx + target.shape[-1]
                input_crop = input_pad[:,:,start_idx:stop_idx]
                target_crop = target[:,:,start_idx:stop_idx]
                               
                output = model(input_crop, c)            # forward pass
                
                loss = criterion(output, target_crop)        # compute loss
                loss.backward()                         # compute gradients
                optimizer.step()                        # update weights

                train_loss += loss.item()

            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)}')
            writer.add_scalar('train_loss', train_loss, global_step=e)
            writer.flush()       

            #################################### Validation Loop #########################################

            model.eval()                  

            for step, (input, target) in enumerate(valid_loader):          # iterate over batches
                print(f"Validation step:{step}", end='\r')
            
                c = torch.tensor([0.0, 0.0]).view(1,1,-1)

                input = input.to(args.device)
                target = target.to(args.device)
                c = c.to(args.device)

                output = model(input, c)
                loss = criterion(output, target)
                valid_loss += loss.item() * input.size(0)

            valid_loss /= len(valid_loader)
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss}')
            writer.add_scalar('valid_loss', valid_loss, global_step=e)
            global_step += 1

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving model ...')
                min_valid_loss = valid_loss

                save_to = f'checkpoints/tcn{format(timestamp)}.pth'
                torch.save(model.state_dict(), save_to)         

            scheduler.step()            # update learning rate
            writer.flush()

        prof.export_chrome_trace(f"logs/prof.pt.trace{format(timestamp)}.json") 
        writer.add_graph(model, input_to_model=input, verbose=True)
        writer.close()

if __name__ == "__main__":    
    main()
