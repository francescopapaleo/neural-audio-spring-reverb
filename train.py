from tcn import TCN
from data import SpringDataset
from pathlib import Path
from datetime import datetime

import argparse
import torch
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram, Loudness
import auraloss
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.monitor import TensorboardEventHandler, register_event_handler

torch.manual_seed(42)
torch.cuda.empty_cache()
    
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='../plate-spring/spring/', help='dataset')
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
parser.add_argument('--crop', type=int, default=3200)
args = parser.parse_args()

def training(n_epochs=args.n_epochs, batch_size=args.batch_size, device=args.device, **args):
    print("Initializing Training Process..", end='\n\n')
    
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    sample_rate = 16000
    global crop 
    crop = 3200

    if args.device is None: 
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Torch version: {torch.__version__} ------ Selected Device: {args.device}')
    print(f'Sample Rate: {sample_rate} Hz ------  Crop Lenght:{crop} samples', end='\n\n')
    
    dataset = SpringDataset(root_dir=args.data_dir, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=0, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=0, shuffle=False, drop_last=True)

    hparams_dict = {                                                # hyperparameters
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'lr': 0.01,
        'sched_gamma': 0.1,
        'n_inputs': 1,
        'n_outputs': 1,
        'n_blocks': 10,
        'kernel_size': 11,
        'n_channels': 64,
        'dilation_growth': 4,
        'cond_dim': 0,
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')            # timestamp for tensorboard

    writer = SummaryWriter(f'runs/tcn_{args.n_epochs}_{args.batch_size}_{timestamp}')  # tensorboard writer
    register_event_handler(TensorboardEventHandler(writer))                            # register event handler
    writer.add_hparams(hparams_dict, {})                                               # Log hyperparameters to TensorBoard

    model = TCN(                                                        # instantiate model     
        n_inputs = hparams_dict['n_inputs'], 
        n_outputs = hparams_dict['n_outputs'], 
        n_blocks = hparams_dict['n_blocks'],
        kernel_size = hparams_dict['kernel_size'],
        n_channels = hparams_dict['n_channels'], 
        dilation_growth = hparams_dict['dilation_growth'],
        cond_dim = hparams_dict['cond_dim'],
    ).to(args.device)
        
    rf = model.compute_receptive_field()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {params*1e-3:0.3f} k")
    print(f"Receptive field: {rf} samples or {(rf / sample_rate)*1e3:0.1f} ms", end='\n\n')       
    
    criterion = auraloss.freq.STFTLoss().to(args.device)                                # loss function       

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)                           # optimizer
    
    ms1 = int(n_epochs * 0.8)
    ms2 = int(n_epochs * 0.95)
    milestones = [ms1, ms2]
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones,
    gamma=0.1,
    verbose=False,
    )

    print("Training Loop", end='\n\n')    
    
    min_valid_loss = np.inf
    c = torch.tensor([0.0, 0.0]).view(1,1,-1)                                          # dummy condition tensor

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
            stop_idx = start_idx + crop
            if stop_idx > input.shape[-1]:
                stop_idx = input.shape[-1]
                start_idx = stop_idx - crop
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

        
        model.eval()
        with torch.no_grad():
            
            for step, (input, target) in enumerate(valid_loader):
                input = input.to(args.device)               # move input and target to device
                target = target.to(args.device)
                c = c.to(args.device)

                start_idx = rf                              # crop input and target
                stop_idx = start_idx + crop
                if stop_idx > input.shape[-1]:
                    stop_idx = input.shape[-1]
                    start_idx = stop_idx - crop
                input_crop = input[:, :, start_idx:stop_idx]
                target_crop = target[:, :, start_idx:stop_idx]     

                output = model(input_crop, c)               # forward pass

                loss = criterion(output, target_crop)          # compute loss
                
                valid_loss += loss.item()                   # accumulate loss
                avg_valid_loss = valid_loss / len(valid_loader)

                writer.add_scalar('Validation loss', avg_valid_loss, global_step = epoch * len(valid_loader))

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) Saving model ...')
                save_to = f'runs/tcn_{n_epochs}_{batch_size}_{timestamp}/tcn_ckpt_{n_epochs}.pth'
                torch.save(model.state_dict(), save_to)
                min_valid_loss = valid_loss
    
        scheduler.step()
        current_lr = scheduler.get_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, epoch)  # log learning rate to tensorboard  
            
        print(f'Epoch {epoch +1} \t\t Validation Loss: {avg_valid_loss:.6f}, \t\t Training Loss: {avg_train_loss:.6f}', end='\r')
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        
    writer.add_graph(model, input_to_model=input, verbose=False)
    writer.flush()
    writer.close()
    
    print('Training Completed!')
    
if __name__ == "__main__":    
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [64, 128, 256]
    num_epochs = [10, 20, 30]

    # Loop over all combinations
    for lr in learning_rates:
        for bs in batch_sizes:
            for ne in num_epochs:
                # Update your arguments or however you're setting hyperparameters
                args.lr = lr
                args.batch_size = bs
                args.n_epochs = ne

                # Print the current hyperparameters for clarity
                print(f"Training with lr={lr}, batch_size={bs}, num_epochs={ne}")

                # Call your training function
                training()
