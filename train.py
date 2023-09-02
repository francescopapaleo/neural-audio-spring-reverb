import torch
import torchaudio
import auraloss
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from src.dataload.egfxset import load_egfxset
from src.dataload.springset import load_springset
from src.train_conf import train_conf
from src.models.helpers import select_device, initialize_model, save_model_checkpoint, load_model_checkpoint
from src.default_args import parse_args

def main():
    args = parse_args()
    torch.manual_seed(42)
    
    device = select_device(args.device)

    torch.backends.cudnn.deterministic = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    torch.cuda.empty_cache()

    # If there's a checkpoint, load it first
    if args.checkpoint is not None:
        model, optimizer_state_dict, scheduler_state_dict, hparams, rf, params = load_model_checkpoint(device, args.checkpoint, args)
        
        if args.lr is not None:
            hparams['lr'] = args.lr
        if args.batch_size is not None:
            hparams['batch_size'] = args.batch_size

        if optimizer_state_dict is not None:
            optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
            optimizer.load_state_dict(optimizer_state_dict)

        if scheduler_state_dict is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)
            scheduler.load_state_dict(scheduler_state_dict)
        
    else:
        # Else, get configuration from the args
        print(f"Using configuration {args.conf}")
        sel_conf = next((c for c in train_conf if c['conf_name'] == args.conf), None)
        if sel_conf is None:
            raise ValueError('Configuration not found')
        hparams = {
            **sel_conf,
            'state_epoch': None,
            'avg_valid_loss': None,
        }        
        model, rf, params = initialize_model(device, hparams)
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])   
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        
    # Initialize Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.log_dir) / f"{hparams['conf_name']}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)

    # print(model)

    # Define loss function and optimizer
    mae = torch.nn.L1Loss().to(device)
    dc = auraloss.time.DCLoss().to(device)
    esr = auraloss.time.ESRLoss().to(device)
    mrstft =  auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=hparams['sample_rate'],
        perceptual_weighting=True,
        ).to(device)

    # Define individual loss choices
    criterion_choices = {
        'mae': mae,
        'mrstft': mrstft,
        'esr': esr,
        'dc': dc
    }

    # Get the chosen criterion from hparams or default to 'mae'
    criterion = criterion_choices.get(hparams['criterion'], mrstft)

    # Print the selected criterion name
    print(f"Using loss: {criterion.__class__.__name__}")
    
    # Load data
    if args.dataset == 'egfxset':
        train_loader, valid_loader, _ = load_egfxset(args.data_dir, batch_size=hparams['batch_size'], num_workers=args.num_workers)
    elif args.dataset == 'springset':
        train_loader, valid_loader, _ = load_springset(args.data_dir, batch_size=hparams['batch_size'], num_workers=args.num_workers)
    else:
        raise ValueError('Dataset not found, options are: egfxset or springset')


    # Initialize minimum validation loss with infinity
    if hparams.get('avg_valid_loss', None) is not None:
        min_valid_loss = hparams['avg_valid_loss']
    else:
        min_valid_loss = np.inf

    # Correct epoch calculation
    if hparams.get('state_epoch', None) is not None:
        state_epoch = hparams['state_epoch']
        current_epoch = state_epoch
        max_epochs = args.max_epochs
    else:
        state_epoch = 0
        current_epoch = 0
        max_epochs = args.max_epochs
    
    print(f"Training model for {max_epochs} epochs, current epoch {current_epoch}")
    avg_train_loss = np.inf 
    avg_valid_loss = np.inf
    patience_count = 0

    try:
        for epoch in range(current_epoch, max_epochs):
            train_loss = 0.0

            model.train()
            for batch_idx, (dry, wet) in enumerate(train_loader):
                optimizer.zero_grad() 
                # input shape: [batch, channel, lenght]
                input = dry.to(device)                             
                target = wet.to(device)
                
                output = model(input)
                
                # Pre-emphasis filter
                pre_emphasis = hparams.get('pre_emphasis', None)
                if pre_emphasis is not None:
                    output = torchaudio.functional.preemphasis(output, float(pre_emphasis))

                loss = criterion(output, target)

                loss.backward()                             
                optimizer.step()
         
                train_loss += loss.item()
                
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('train/learning_rate', lr, global_step=current_epoch)

            avg_train_loss = train_loss / len(train_loader)
            writer.add_scalar('train/loss_train', avg_train_loss, global_step=epoch)

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for step, (dry, wet) in enumerate(valid_loader):
                    input = dry.to(device)
                    target = wet.to(device)
                
                    output = model(input)
                    
                    # Pre-emphasis filter
                    pre_emphasis = hparams.get('pre_emphasis', None)
                    if pre_emphasis is not None:
                        output = torchaudio.functional.preemphasis(output, float(pre_emphasis))
                    
                    loss = criterion(output, target)

                    valid_loss += loss.item()                   
                avg_valid_loss = valid_loss / len(valid_loader)    
    
            writer.add_scalar('train/loss_valid', avg_valid_loss, global_step=current_epoch)
            
            scheduler.step(avg_valid_loss)

            # Save the model if it improved
            if avg_valid_loss < min_valid_loss:
                print(f"Epoch {epoch}: Loss improved from {min_valid_loss:4f} to {avg_valid_loss:4f} - > Saving model", end="\r")
                min_valid_loss = avg_valid_loss
                save_model_checkpoint(
                    model, hparams, optimizer, scheduler, current_epoch, timestamp, avg_valid_loss, args
                )
            else:
                patience_count += 1
                if args.early_stop is not None and patience_count >= args.early_stop:
                    print(f"Epoch {epoch}: Loss did not improve for {args.early_stop} epochs, stopping training")
                    break 
            
            current_epoch += 1

    except KeyboardInterrupt:
        print("\nTraining manually stopped by user. Processing the results...")

    finally:
        final_train_loss = avg_train_loss
        final_valid_loss = avg_valid_loss
        writer.add_hparams(hparams, {'train/final': final_train_loss, 'train/final_valid': final_valid_loss})
        print(f"Epoch {epoch}, final validation loss: {final_valid_loss}")    

        output = torchaudio.functional.highpass_biquad(output, hparams["sample_rate"], 20)
        target = torchaudio.functional.highpass_biquad(target, hparams["sample_rate"], 20)

        target = target.view(-1).unsqueeze(0).cpu()
        output = output.view(-1).unsqueeze(0).cpu()

        target /= torch.max(torch.abs(target))                
        output /= torch.max(torch.abs(output))

        save_out = f"{args.audio_dir}/output_{hparams['conf_name']}.wav"
        torchaudio.save(save_out, output, hparams['sample_rate'])

        save_target = f"{args.audio_dir}/target_{hparams['conf_name']}.wav"
        torchaudio.save(save_target, target, hparams['sample_rate'])

        writer.close()

if __name__ == "__main__":

    main()
