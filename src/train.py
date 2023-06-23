# train.py

import torch
import auraloss
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from utils.helpers import load_data, initialize_model, save_model_checkpoint, parse_args

torch.manual_seed(42)            

def training_loop(model, criterion, esr, optimizer, train_loader, device, writer, global_step):
    """Train the model for one epoch"""
    train_loss = 0.0
    train_metric = 0.0
    model.train()
    c = torch.tensor([0.0, 0.0]).view(1,1,-1).to(device)
    for batch_idx, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        input, target = input.to(device), target.to(device)
        output = model(input, c)
        loss = criterion(output, target)             
        metric = esr(output, target)
        loss.backward()                             
        optimizer.step()                            
        train_loss += loss.item()
        train_metric += metric.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_train_metric = train_metric / len(train_loader)

    writer.add_scalar('training/stft', avg_train_loss, global_step = global_step)
    writer.add_scalar('training/esr', avg_train_metric, global_step = global_step)

    return model, avg_train_loss, avg_train_metric

def validation_loop(model, criterion, esr, valid_loader, device, writer, global_step):
    """Validation loop for one epoch"""
    model.eval()
    valid_loss = 0.0
    valid_metric = 0.0
    c = torch.tensor([0.0, 0.0]).view(1,1,-1).to(device) 
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_loader):
            input, target = input.to(device), target.to(device)
            output = model(input, c)
            loss = criterion(output, target)
            metric = esr(output, target)
            valid_loss += loss.item()                   
            valid_metric += metric.item()

    avg_valid_loss = valid_loss / len(valid_loader)    
    avg_valid_metric = valid_metric / len(valid_loader)

    writer.add_scalar('validation/stft', avg_valid_loss, global_step = global_step)
    writer.add_scalar('validation/esr', avg_valid_metric, global_step = global_step)
    
    return avg_valid_loss, avg_valid_metric


def main():
    # Parse command line arguments
    args = parse_args()

    # Define hyperparameters
    hparams = {
        'n_inputs': 1,
        'n_outputs': 1,
        'n_blocks': 10,
        'kernel_size': 15,
        'n_channels': 64,
        'dilation_growth': 2,
        'cond_dim': 0,
    }

    # Define loss function and optimizer
    device = torch.device(args.device)
    criterion = auraloss.freq.STFTLoss().to(device)  
    esr = auraloss.time.ESRLoss().to(device)

    # Initialize model
    model = initialize_model(device, "TCN", hparams)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ms1 = int(args.n_epochs * 0.8)
    ms2 = int(args.n_epochs * 0.95)
    milestones = [ms1, ms2]
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones,
        gamma=0.1,
        verbose=False,
    )

    # Initialize Tensorboard writer
    log_dir = Path(args.logdir) / f"tcn_{args.n_epochs}_{args.batch_size}_{args.lr}"
    writer = SummaryWriter(log_dir=log_dir)

    # Load data
    train_loader, valid_loader, _ = load_data(args.datadir, args.batch_size)
    
    # Initialize minimum validation loss with infinity
    min_valid_loss = np.inf

    # Loop through each epoch
    for epoch in range(args.n_epochs):
        # Train model
        model, train_loss, train_metric = training_loop(
            model, criterion, esr, optimizer, train_loader, device, writer, epoch)
        
        # Validate model
        valid_loss, valid_metric = validation_loop(
            model, criterion, esr, valid_loader, device, writer, epoch)

        # Update learning rate
        scheduler.step()

        # Save the model if it improved
        if valid_loss < min_valid_loss:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            min_valid_loss = valid_loss
            save_model_checkpoint(
                model, hparams, criterion, optimizer, scheduler, args.n_epochs, args.batch_size, args.lr, timestamp
            )

if __name__ == "__main__":
    main()