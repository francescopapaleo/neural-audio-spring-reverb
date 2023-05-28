# python train.py --batch_size 4 --epochs 10 --device cpu
import argparse
from tcn import TCN
from utils.data import SpringDataset
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.monitor import TensorboardEventHandler, register_event_handler
from torchsummary import summary

import torchaudio.transforms as T
import auraloss

torch.backends.cudnn.benchmark = True

def main():
    print(f'Torch version: {torch.__version__}')
    torch.set_default_dtype(torch.float32)  
    torch.set_default_tensor_type(torch.FloatTensor) 

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/tcn_train_{}'.format(timestamp))
    register_event_handler(TensorboardEventHandler(writer))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/plate-spring/spring/', 
                    help='default dataset folder')
    parser.add_argument('--models_dir', type=str, default='./models', 
                    help='folder to store state_dict after training and load for eval or inference')
    parser.add_argument('--batch_size', type= int)
    parser.add_argument('--epochs', type= int)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    device = args.device
    torch.device('cpu')

    dataset = SpringDataset(root_dir=args.data_dir, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, dtype=torch.float32)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, shuffle=False, dtype=torch.float32)
    print(f'Batch size is : {batch_size}')

    model = TCN(
        n_inputs=1, 
        n_outputs=1, 
        n_blocks=10, 
        kernel_size=1, 
        n_channels=16, 
        dilation_growth=2, 
        cond_dim=0,
        dtype=torch.float32
    )
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("## Loss Function")
    stft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048],
        win_lengths=[32, 128, 512, 2048],
        hop_sizes=[16, 64, 256, 1024],
        sample_rate=16000)
    
    criterion = stft
    min_valid_loss = np.inf

    print(criterion, model.named_parameters)
    
    model.train()
    for e in range(epochs):
        print(f'Epoch {e+1}/{epochs} \t\t Training')
        train_loss = 0.0

        for step, (input, target) in tqdm(enumerate(train_loader), unit='batch', total=int(len(valid_loader))):
            input.to(device, dtype=torch.float32)
            target.to(device, dtype=torch.float32)
            c = torch.ones_like(input).to(device, dtype=torch.float32)

            optimizer.zero_grad()
            output = model(input, c)
            
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)}')
        writer.add_scalar('train_loss', train_loss, global_step=e)     
        writer.flush()

        valid_loss = 0.0
        model.eval()
        for step, (input, target) in tqdm(enumerate(valid_loader), unit='batch', total=int(len(valid_loader))):
            print(f'Epoch {e+1}/{epochs} \t\t Validation')
                                          
            input.to(device, dtype=torch.float32)
            target.to(device, dtype=torch.float32)
            c = torch.ones_like(input).to(device, dtype=torch.float32)
            output = model(input, c)

            loss = criterion(output, target)
            valid_loss = loss.item() * input.size(0)

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
        writer.add_scalar('valid_loss', valid_loss, global_step=e)

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            writer.flush()

            save_to = f'runs/tcn.pth{format(timestamp)}'
            torch.save(model.state_dict(), save_to)         
    
    writer.add_graph(model, input_to_model=input, verbose=True)
    writer.close()


if __name__ == "__main__":
    
    main()
