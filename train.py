# train.py

import torch
import auraloss
import numpy as np
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from src.helpers import load_data, select_device, initialize_model, save_model_checkpoint
from configurations import parse_args, configs

torch.manual_seed(42)
torch.cuda.empty_cache()            

def training_loop(model, criterion_a, criterion_b, alpha, optimizer, train_loader, device, writer, global_step):
    """Train the model for one epoch"""
    train_loss = 0.0

    model.train()
    c = torch.tensor([0.0, 0.0]).view(1,1,-1).to(device)
    for batch_idx, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        input, target = input.to(device), target.to(device)
        # print(input.shape)
        output = model(input, c)
        
        loss_a = criterion_a(output, target)
        loss_b = criterion_b(output, target)             
        loss = alpha * loss_a + (1 - alpha) * loss_b
        
        loss.backward()                             
        optimizer.step()                            
        train_loss += loss.item()
        
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('training/learning_rate', lr, global_step=global_step)

    avg_train_loss = train_loss / len(train_loader)
    
    writer.add_scalar('training/loss', avg_train_loss, global_step = global_step)


    return model, avg_train_loss

def validation_loop(model, criterion_a, criterion_b, alpha, valid_loader, device, writer, global_step):
    """Validation loop for one epoch"""
    model.eval()
    valid_loss = 0.0
    c = torch.tensor([0.0, 0.0]).view(1,1,-1).to(device) 
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_loader):
            input, target = input.to(device), target.to(device)
            output = model(input, c)

            loss_a = criterion_a(output, target)
            loss_b = criterion_b(output, target)             
            loss = alpha * loss_a + (1 - alpha) * loss_b

            valid_loss += loss.item()                   

    avg_valid_loss = valid_loss / len(valid_loader)    
    
    writer.add_scalar('validation/loss', avg_valid_loss, global_step = global_step)
    
    return avg_valid_loss



def main():
    args = parse_args()

    device = select_device(args.device)

     # Find the configuration in the list
    print(f"Using configuration {args.config}")
    sel_config = next((c for c in configs if c['conf_name'] == args.config), None)
    if sel_config is None:
        raise ValueError('Configuration not found')
    hparams = sel_config

    # Initialize model
    model, rf, params = initialize_model(device, hparams)

    # Initialize Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.logdir) / f"train/{hparams['conf_name']}_{args.n_epochs}_{args.batch_size}_{args.lr}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    
    writer.add_text('model_summary', str(model), global_step=0)

    torch.cuda.empty_cache()

    # Define loss function and optimizer
    mae = torch.nn.L1Loss().to(device)
    esr = auraloss.time.ESRLoss().to(device)
    mrstft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048],
        win_lengths=[32, 128, 512, 2048],
        hop_sizes=[16, 64, 256, 1024]).to(device)

    alpha = 0.5
    criterion_a = mae
    criterion_b = esr

    if criterion_a == mrstft and criterion_b == esr:
        criterion_str = "mrstft+esr"
    elif criterion_a == mrstft and criterion_b == mae:
        criterion_str = "mrstft+mae"
    elif criterion_a == mae and criterion_b == esr:
        criterion_str = "mae+esr"
    else:
        criterion_str = "unknown"

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

    # Load data
    train_loader, valid_loader, _ = load_data(args.datadir, args.batch_size)
    
    # Initialize minimum validation loss with infinity
    min_valid_loss = np.inf
    patience_counter = 0

    hparams.update({
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'receptive_field': rf,
        'params': params,
        'criterion': criterion_str,
    })

    print(f"Training model for {args.n_epochs} epochs, with batch size {args.batch_size} and learning rate {args.lr}")
    for epoch in range(args.n_epochs):
        # Train model
        model, train_loss = training_loop(
            model, criterion_a, criterion_b, alpha, optimizer, train_loader, device, writer, epoch)
        
        # Validate model
        valid_loss = validation_loop(
            model, criterion_a, criterion_b, alpha, valid_loader, device, writer, epoch)

        # Update learning rate
        scheduler.step()

        # Save the model if it improved
        if valid_loss < min_valid_loss:
            print(f"Validation loss improved from {min_valid_loss} to {valid_loss}. Saving model.")
            
            min_valid_loss = valid_loss
            save_model_checkpoint(
                model, hparams, criterion_str, optimizer, scheduler, args.n_epochs, args.batch_size, args.lr, timestamp
            )
            patience_counter = 0  # reset the counter if performance improved
        else:
            patience_counter += 1  # increase the counter if performance did not improve

        # Early stopping if performance did not improve after n epochs
        if patience_counter >= 20:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement in validation loss.")
            break

        final_train_loss = train_loss
        final_valid_loss = valid_loss
        writer.add_hparams(hparams, {'Final Training ESR': final_train_loss, 'Final Validation ESR': final_valid_loss})

        writer.close()

if __name__ == "__main__":
    main()