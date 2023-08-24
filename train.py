import torch
import torchaudio
import torchaudio.functional as F
import auraloss
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from src.helpers import load_data, select_device, initialize_model, save_model_checkpoint, load_model_checkpoint
from configurations import parse_args, configs

def main():
    args = parse_args()
    torch.manual_seed(42)
    
    device = select_device(args.device)

    torch.backends.cudnn.deterministic = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    torch.cuda.empty_cache()

    # If there's a checkpoint, load it first
    if args.checkpoint is not None:
        model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params = load_model_checkpoint(device, args.checkpoint, args)
        if optimizer_state_dict is not None:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  
            optimizer.load_state_dict(optimizer_state_dict)
        if scheduler_state_dict is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            scheduler.load_state_dict(scheduler_state_dict)
    else:
        # Else, get configuration from the args
        print(f"Using configuration {args.config}")
        sel_config = next((c for c in configs if c['conf_name'] == args.config), None)
        if sel_config is None:
            raise ValueError('Configuration not found')
        hparams = sel_config
        model, rf, params = initialize_model(device, hparams, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Initialize Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.logdir) / f"train/{hparams['conf_name']}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    torch.cuda.empty_cache()

    print(model)

    # Define loss function and optimizer
    mae = torch.nn.L1Loss().to(device)
    mse = torch.nn.MSELoss().to(device)
    dc = auraloss.time.DCLoss().to(device)
    esr = auraloss.time.ESRLoss().to(device)
    mrstft =  auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048],
        win_lengths=[32, 128, 512, 2048],
        hop_sizes=[16, 64, 256, 1024],
        sample_rate=args.sample_rate,
        perceptual_weighting=True,
        ).to(device)
    
    criterion = hparams['criterion']
    alpha = 1.0

    # Load data
    train_loader, valid_loader, _ = load_data(args.datadir, args.batch_size)
    
    # Initialize minimum validation loss with infinity
    min_valid_loss = np.inf

    hparams.update({
        'n_epochs': args.n_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'receptive_field': rf,
        'params': params,
        'sample_rate': args.sample_rate,
    })

    start_epoch = last_epoch + 1 if args.checkpoint is not None else 0

    print(f"Training model for {args.n_epochs} epochs, with batch size {args.batch_size} and learning rate {args.lr}")
    try:
        for epoch in range(start_epoch, start_epoch + args.n_epochs):
            train_loss = 0.0

            # Training
            model.train()
            c = torch.tensor([0.0, 0.0]).view(1,1,-1).to(device)
            for batch_idx, (dry, wet) in enumerate(train_loader):
                optimizer.zero_grad() 
                input = dry.to(device)          # shape: [batch, channel, seq]
                target = wet.to(device)
                
                output = model(input)
                
                # output = torchaudio.functional.preemphasis(output, 0.95)
                loss_1 = mae(output, target)
                loss_2 = mrstft(output, target)
                loss = loss_1 + (alpha * loss_2)

                loss.backward()                             
                optimizer.step()
         
                train_loss += loss.item()
                
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('training/learning_rate', lr, global_step=epoch)

            avg_train_loss = train_loss / len(train_loader)
            writer.add_scalar('training/loss', avg_train_loss, global_step=epoch)
            
            # Validation
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for step, (dry, wet) in enumerate(valid_loader):
                    input = dry.to(device)
                    target = wet.to(device)
                
                    output = model(input)
                    
                    # output = torchaudio.functional.preemphasis(output, 0.95)
                    loss_1 = mae(output, target)
                    loss_2 = mrstft(output, target)
                    loss = loss_1 + (alpha * loss_2)

                    valid_loss += loss.item()                   
                avg_valid_loss = valid_loss / len(valid_loader)    
    
            writer.add_scalar('validation/loss', avg_valid_loss, global_step=epoch)
            
            # Update learning rate
            scheduler.step(avg_valid_loss)

            # Save the model if it improved
            if avg_valid_loss < min_valid_loss:
                print(f"Epoch {epoch}: Loss improved from {min_valid_loss:4f} to {avg_valid_loss:4f} - > Saving model", end="\r")
                min_valid_loss = avg_valid_loss
                save_model_checkpoint(
                    model, hparams, optimizer, scheduler, epoch, timestamp, avg_valid_loss, args
                )

    finally:
        final_train_loss = avg_train_loss
        final_valid_loss = avg_valid_loss
        writer.add_hparams(hparams, {'Final Training Loss': final_train_loss, 'Final Validation Loss': final_valid_loss})
        print(f"Final Validation Loss: {final_valid_loss}")    

        inp = input.view(-1).unsqueeze(0).cpu()
        tgt = target.view(-1).unsqueeze(0).cpu()
        out = output.view(-1).unsqueeze(0).cpu()

        inp /= torch.max(torch.abs(inp))
        tgt /= torch.max(torch.abs(tgt))                
        out /= torch.max(torch.abs(out))

        save_in = f"{log_dir}/inp_{hparams['conf_name']}.wav"
        torchaudio.save(save_in, inp, args.sample_rate, bits_per_sample=24)

        save_out = f"{log_dir}/out_{hparams['conf_name']}.wav"
        torchaudio.save(save_out, out, args.sample_rate, bits_per_sample=24)

        save_target = f"{log_dir}/tgt_{hparams['conf_name']}.wav"
        torchaudio.save(save_target, tgt, args.sample_rate, bits_per_sample=24)

    writer.close()

if __name__ == "__main__":

    main()
