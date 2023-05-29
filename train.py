# python train.py --batch_size 4 --epochs 10 --device cpu
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
parser.add_argument('--models_dir', type=str, default='./models', help='state dict')
parser.add_argument('--batch_size', type= int, default=2)
parser.add_argument('--epochs', type= int, default=10)
parser.add_argument('--device', type=str, default='cuda:0')

def main():

    print("---------------------------------------------------------------------")
    print("_____________Temporal Convolutional Neural Network___________________")
    print("---------------------------------------------------------------------")
    print("..................Initializing training process......................")
    
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size

    # torch.set_default_dtype(torch.float32)  
    # torch.set_default_tensor_type(torch.FloatTensor) 

    print(f'Torch version: {torch.__version__}'), print(f'Cuda available: {torch.cuda.is_available()}')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    writer = SummaryWriter('runs/tcn_train_{}'.format(timestamp))
    register_event_handler(TensorboardEventHandler(writer))

    dataset = SpringDataset(root_dir=args.data_dir, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=0, shuffle=False)

    hparams = {"channel_growth": 1,
                "kernel_size": 3,
                "dilation_growth": 2,
                "causal": False,
                "grouped": False,
                "nparams": 0,
                "skip_connection": False,
    }

    model = TCN(nparams=1,
        n_inputs=1, 
        n_outputs=1, 
        n_blocks=10,
        kernel_size=9,
        dilation_growth=2,
        channel_growth=2,
        channel_width=32,
        stack_size=10,
        num_examples=4,
        save_dir=None,
        n_channels=2,
    )
    model.to(args.device)
               
    criterion = torch.nn.MSELoss().to(args.device)   

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # optimizer
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1, verbose=True)
    
    min_valid_loss = np.inf                 # initialize min_valid_loss 

    # Profiling
    with profiler.profile(use_cuda=True, use_kineto=True, record_shapes=True) as prof:

        ################################## Training Loop ###########################################

        model.train()

        for e in tqdm(range(epochs), unit='epoch', total=epochs, position=0, leave=True):      # iterate over epochs
            train_loss = 0.0

            for step, (input, target) in enumerate(train_loader):       # iterate over batches
                print(f"step:{step}", end='\r')

                c = torch.ones_like(input)

                input = input.to(args.device)           # move everything to device
                target = target.to(args.device)
                c = c.to(args.device)
                
                output = model(input, c)                # forward pass
                
                # Padding to match the output and target shapes
                output = F.pad(output, (0, target.size(2) - output.size(2)))  
                
                loss = criterion(output, target)    # compute loss
                optimizer.zero_grad()            # clear gradients
                loss.backward()             # compute gradients
                optimizer.step()            # update weights

                train_loss += loss.item()

            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)}')
            writer.add_scalar('train_loss', train_loss, global_step=e)
            writer.flush()       

            #################################### Validation Loop #########################################

            model.eval()                  
            valid_loss = 0.0

            for step, (input, target) in enumerate(valid_loader):          # iterate over batches
                print(f"Validation step:{step}", end='\r')
                
                c = torch.ones_like(input)

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
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss

                save_to = f'runs/tcn.pth{format(timestamp)}'
                torch.save(model.state_dict(), save_to)         

            scheduler.step()            # update learning rate
            writer.flush()

        writer.add_graph(model, input_to_model=input, verbose=True)
        writer.close()

if __name__ == "__main__":    
    main()
