import argparse
from tcn import TCN
from data import SpringDataset
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.monitor import TensorboardEventHandler, register_event_handler
import torch.nn.functional as F
import torch
import torchaudio.transforms as T
import auraloss

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/plate-spring/spring/', help='dataset')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='state dict')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--device', type=str)
parser.add_argument('--crop', type=int, default=3200)

def training():
    print("#-----------------------------------------------------------------------#")
    print("                     Initializing training process")  
    print("-------------------------------------------------------------------------")
    print("")
    
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    sample_rate = 16000
    global crop_len 
    crop_len = args.crop

    if args.device is None: 
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Torch version: {torch.__version__} ------ Selected Device: {args.device}')
    print(f'Sample Rate: {sample_rate} Hz ------  Crop Lenght: {crop_len} samples')
    print("-------------------------------------------------------------------------")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    writer = SummaryWriter('logs/tcn_{}'.format(timestamp))
    register_event_handler(TensorboardEventHandler(writer))
    
    dataset = SpringDataset(root_dir=args.data_dir, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=4, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=4, shuffle=False, drop_last=True)

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
    print(f"Receptive field: {rf} samples or {(rf / sample_rate)*1e3:0.1f} ms")       
    print("-------------------------------------------------------------------------")
    
    # define the loss function
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048], 
        win_lengths=[32, 128, 512, 2048],
        hop_sizes=[16, 64, 256, 1024]).to(args.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # optimizer
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50], gamma=0.1, verbose=True)
    
    c = torch.tensor([0.0, 0.0]).view(1,1,-1)

    ############################### Training Loop #################################
    
    running_loss = 0.0              # initialize running_loss
    min_valid_loss = np.inf             # initialize min_valid_loss
    
   
    for e in (range(epochs)):      # iterate over epochs
        model.train()
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

            loss = mrstft(output, target_crop)
            
            loss.backward()  # compute gradients
            optimizer.step()  # update weights

            train_loss += loss.item()
            writer.add_scalar('Batch Loss', loss.item(), global_step=batch_idx+1 + e*len(train_loader))
            writer.flush()

        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)}')
        writer.add_scalar('Training Loss', train_loss, global_step=e)
        writer.flush()    

        #################################### Validation Loop #########################################
        valid_loss = 0.0                                               # initialize valid_loss
        model.eval()
        with torch.no_grad():                
            for step, (input, target) in enumerate(valid_loader):
                print(f"Validation step:{step}", end='\r')

                input = input.to(args.device)
                target = target.to(args.device)
                c = c.to(args.device)

                start_idx = rf
                stop_idx = start_idx + crop_len
                if stop_idx > input.shape[-1]:
                    stop_idx = input.shape[-1]
                    start_idx = stop_idx - crop_len
                input_crop = input[:, :, start_idx:stop_idx]
                target_crop = target[:, :, start_idx:stop_idx]     

                output = model(input_crop, c)

                loss = mrstft(output, target_crop)
                
                valid_loss += loss.item()
        
            valid_loss /= len(valid_loader)
            print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss}')
            writer.add_scalar('valid_loss', valid_loss, global_step=e)

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving model ...')
                min_valid_loss = valid_loss

                save_to = 'checkpoints/tcn{}.pth'.format(timestamp)
                torch.save(model.state_dict(), save_to)         
                writer.flush()
        
        scheduler.step()
    writer.add_graph(model, input_to_model=input, verbose=False)
    writer.flush()
    writer.close()

    print('                         Finished Training')
    print("#-----------------------------------------------------------------------#")
    
if __name__ == "__main__":    
    training()


'''
Loss Functions:
@inproceedings{steinmetz2020auraloss,
    title={auraloss: {A}udio focused loss functions in {PyTorch}},
    author={Steinmetz, Christian J. and Reiss, Joshua D.},
    booktitle={Digital Music Research Network One-day Workshop (DMRN+15)},
    year={2020}
}
'''