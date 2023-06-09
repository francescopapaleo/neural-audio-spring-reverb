""" Training script
Train a TCN model on the training set and save the results with Tensorboard.
"""

import torch
import auraloss
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from TCN import TCNBase
from data import SpringDataset
from datetime import datetime
from argparse import ArgumentParser

torch.cuda.empty_cache()
torch.manual_seed(42)            

   
def training(data_dir, device, sample_rate, n_epochs, batch_size, lr, crop):

    print("Initializing Training Process..", end='\n\n')
    if device is None: 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Torch version: {torch.__version__} ------ Selected Device: {device}')
    print(f'Sample Rate: {sample_rate} Hz ------  Crop Lenght:{crop} samples', end='\n\n')
    
    dataset = SpringDataset(root_dir=data_dir, split='train')
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, valid = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train, batch_size, num_workers=0, shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size, num_workers=0, shuffle=False, drop_last=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'runs/train01C/tcn_{n_epochs}_{batch_size}_{lr}_{timestamp}')
    hparams = ({
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'l_rate': lr,
        'sched_gamma': 0.1,
        'n_inputs': 1,
        'n_outputs': 1,
        'n_blocks': 10,
        'kernel_size': 15,
        'n_channels': 64,
        'dilation_growth': 2,
        'cond_dim': 0,
        })                     
    
    model = TCNBase(                                                    # instantiate model     
        n_inputs = hparams['n_inputs'], 
        n_outputs = hparams['n_outputs'], 
        n_blocks = hparams['n_blocks'],
        kernel_size = hparams['kernel_size'],
        n_channels = hparams['n_channels'], 
        dilation_growth = hparams['dilation_growth'],
        cond_dim = hparams['cond_dim'],
    ).to(device)

    model_summary = str(model).replace( '\n', '<br/>').replace(' ', '&nbsp;')
    writer.add_text("model", model_summary)
    writer.flush()

    rf = model.compute_receptive_field()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {params*1e-3:0.3f} k")
    print(f"Receptive field: {rf} samples or {(rf / sample_rate)*1e3:0.1f} ms", end='\n\n')       
    
    criterion = auraloss.freq.STFTLoss().to(device)                 # loss function       
    esr = auraloss.time.ESRLoss().to(device)                        # metric

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)         # optimizer

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
    c = torch.tensor([0.0, 0.0]).view(1,1,-1)           # dummy condition tensor
    metrics = {'training/esr': [],'validation/esr': []}

    for epoch in range(n_epochs):    
        train_loss = 0.0
        train_metric = 0.0
        avg_train_loss = 0.0
        avg_train_metric = 0.0
        
        model.train()
        for batch_idx, (input, target) in enumerate(train_loader):
            global_step = (epoch * len(train_loader)) + batch_idx
            optimizer.zero_grad()

            input = input.to(device)                   # move input and target to device
            target = target.to(device)
            c = c.to(device)

            input_pad = torch.nn.functional.pad(input, (rf-1, 0))
            target_pad = torch.nn.functional.pad(target, (rf-1, 0))
            input_crop = input_pad
            target_crop = target_pad

            output = model(input_crop, c)                   # forward pass

            loss = criterion(output, target_crop)              # compute loss
            metric = esr(output, target_crop)

            loss.backward()                                 # compute gradients
            optimizer.step()                                # update weights
            
            train_loss += loss.item()
            train_metric += metric.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_metric = train_metric / len(train_loader)

        writer.add_scalar('training/stft', avg_train_loss, global_step = global_step)
        writer.add_scalar('training/esr', avg_train_metric, global_step = global_step)

        model.eval()
        valid_loss = 0.0
        avg_valid_loss = 0.0
        valid_metric = 0.0
        avg_valid_metric = 0.0
            
        with torch.no_grad():
            for step, (input, target) in enumerate(valid_loader):

                input = input.to(device)                    # move input and target to device
                target = target.to(device)
                c = c.to(device)

                input_pad = torch.nn.functional.pad(input, (rf-1, 0))
                target_pad = torch.nn.functional.pad(target, (rf-1, 0))
                input_crop = input_pad
                target_crop = target_pad

                output = model(input_crop, c)               # forward pass

                loss = criterion(output, target_crop)       # compute loss
                metric = esr(output, target_crop)

                valid_loss += loss.item()                   # accumulate loss
                valid_metric += metric.item()

            avg_valid_loss = valid_loss / len(valid_loader)    
            avg_valid_metric = valid_metric / len(valid_loader)

            writer.add_scalar('validation/stft', avg_valid_loss, global_step = global_step)
            writer.add_scalar('validation/esr', avg_valid_metric, global_step = global_step)
            
            if min_valid_loss > avg_valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{avg_valid_loss:.6f}) Saving model ...')
                save_to = f'checkpoints/tcn_{n_epochs}_{batch_size}_{lr}_{timestamp}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'name': f'TCN{n_epochs}_{batch_size}_{lr}_{timestamp}',
                    'hparams': hparams
                }, save_to)

                min_valid_loss = avg_valid_loss
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print("Epoch: {}, Learning Rate: {}".format(epoch, scheduler.get_last_lr()[0]))
            writer.add_scalar('Learning Rate', current_lr, epoch)
        
        metrics['training/esr'].append(avg_train_metric)
        metrics['validation/esr'].append(avg_valid_metric) 
            
        print(f'Epoch {epoch +1} \t\t Validation Loss: {avg_valid_loss:.6f}, \t\t Training Loss: {avg_train_loss:.6f}', end='\r')

    metrics['training/esr'] = np.mean(metrics['training/esr'])
    metrics['validation/esr'] = np.mean(metrics['validation/esr'])            
    
    writer.add_hparams(hparams, metrics)
    
    # Use a dummy input to generate the graph for Tensorboard
    input_shape = (batch_size, 1, crop)
    dummy_input = torch.randn(input_shape, device=device)
    writer.add_graph(model, input_to_model=dummy_input)

    writer.flush()
    writer.close()
    
    print('Training Completed!')
    
if __name__ == "__main__":

    parser = ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='../plate-spring/spring/', help='dataset')
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--sample_rate', type=int, default=16000)

    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--crop', type=int, default=3200)

    args = parser.parse_args()

    lr_list = [0.001]
    bs_list = [8]
    ep_list = [100]

    # Loop over all combinations
    for lr in lr_list:
        for batch_size in bs_list:
            for n_epochs in ep_list:

                print(f"Training with lr={lr}, batch_size={batch_size}, n_epochs={n_epochs}")

                training(args.data_dir, args.device, args.sample_rate, n_epochs, batch_size, lr, args.crop)

