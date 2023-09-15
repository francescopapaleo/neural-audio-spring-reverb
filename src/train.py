import os
import torch
import torchaudio
import torchaudio.functional as F
import auraloss
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from src.data.egfxset import load_egfxset
from src.data.springset import load_springset
from src.data.customset import load_customset
from src.utils.checkpoints import initialize_model, save_model_checkpoint, load_model_checkpoint, parse_config


def train_model(args):
    
    # If there's a checkpoint, resume training and load it first
    if args.checkpoint is not None:
        model, optimizer_state_dict, scheduler_state_dict, config, rf, params = load_model_checkpoint(args)

        if optimizer_state_dict is not None:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            optimizer.load_state_dict(optimizer_state_dict)

        if scheduler_state_dict is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['lr_patience'], verbose=True)
            scheduler.load_state_dict(scheduler_state_dict)
        
    # Else, get configuration from the config file 
    elif args.init is not None:
        config_file_path = Path(args.init)
        if not config_file_path.is_file():
            raise FileNotFoundError(f"The specified configuration file '{config_file_path}' does not exist.")
        config = parse_config(config_file_path)
    
        print(f"Using configuration {config['name']}")
        
        model, rf, params = initialize_model(args.device, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])   
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=config['lr_patience'], verbose=True)
    else:
        raise ValueError("Either a checkpoint or a configuration file must be specified.")

    # Get the timestamp and label for the run      
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sr_tag = str(int(config['sample_rate'] / 1000)) + 'k'
    # label = f"{sr_tag}-{config['name']}-{config['criterion1']}+{config['criterion2']}"
    label = f"{config['name']}-{timestamp}-{sr_tag}"

    # Initialize Tensorboard writer
    log_dir = Path(args.log_dir) / f"train/{label}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Define loss function
    mae = torch.nn.L1Loss().to(args.device)
    dc = auraloss.time.DCLoss().to(args.device)
    esr = auraloss.time.ESRLoss().to(args.device)
    mrstft =  auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 8192],
        hop_sizes=[256, 512, 2048],
        win_lengths=[1024, 2048, 8192],
        scale="mel",
        n_bins=128,
        sample_rate=config['sample_rate'],
        perceptual_weighting=True,
        ).to(args.device)
    smooth = torch.nn.SmoothL1Loss().to(args.device)

    # Define individual loss choices
    criterion_choices = {
        'mae': mae,
        'mrstft': mrstft,
        'esr': esr,
        'dc': dc,
        'smooth': smooth,
    }

    # Get the chosen criterion from the config or set defaults
    criterion1= criterion_choices.get(config['criterion1'], mrstft)
    # if config['criterion2'] is not None:
    criterion2 = criterion_choices.get(config['criterion2'], None)
    print(f"Using losses: {criterion1.__class__.__name__} and {criterion2.__class__.__name__}")
    
    # Load data
    if config['dataset'] == 'egfxset':
        train_loader, valid_loader, _ = load_egfxset(args.data_dir, batch_size=config['batch_size'],
                                                     num_workers=config['num_workers'])
    elif config['dataset'] == 'springset':
        train_loader, valid_loader, _ = load_springset(args.data_dir, batch_size=config['batch_size'],
                                                       num_workers=config['num_workers'])
    elif config['dataset'] == 'customset':
        train_loader, valid_loader, _ = load_customset(args.data_dir, batch_size=config['batch_size'],
                                                       num_workers=config['num_workers'])
    else:
        raise ValueError('Dataset not found, options are: egfxset or springset')

    # Initialize minimum validation loss with infinity
    if config['min_valid_loss'] is None:
        min_valid_loss = np.inf
    else:
        min_valid_loss = config['min_valid_loss']

    current_epoch = config['current_epoch']
    max_epochs = config['max_epochs']
    patience_count = 0
    
    print(f"Training model for {max_epochs} epochs, current epoch {current_epoch}")
    avg_train_loss = np.inf 
    avg_valid_loss = np.inf

    if config['cond_dim'] > 0:
        c0 = config.get('c0', 0.0)
        c1 = config.get('c1', 0.0)
        c = torch.tensor([c0, c1], device=args.device).view(1,1,-1)
    else :
        c = None

    try:
        for epoch in range(current_epoch, max_epochs):
            train_loss = 0.0

            model.train()
            for batch_idx, (dry, wet) in enumerate(train_loader):
                print(f"Epoch {epoch}: Batch {batch_idx}/{len(train_loader)}", end="\r")
                # input shape: [batch, channel, lenght]
                input = dry.to(args.device)                             
                target = wet.to(args.device)

                pred = model(input, c)
                
                # Pre-emphasis filter
                pre_emphasis = config.get('pre_emphasis', None)
                if pre_emphasis is not None:
                    pred = F.preemphasis(pred, float(pre_emphasis))
                
                loss1 = criterion1(pred, target)
                
                if config['criterion2'] is not None:
                    loss2 = criterion2(pred, target)
                else:
                    loss2 = 0.0

                loss = loss1 + loss2

                loss.backward()                             
                optimizer.step()
                optimizer.zero_grad() 
                
                train_loss += loss.item()
                
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('train/learning_rate', lr, global_step=current_epoch)

            avg_train_loss = train_loss / len(train_loader)
            writer.add_scalar('train/loss_train', avg_train_loss, global_step=epoch)

            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for step, (dry, wet) in enumerate(valid_loader):
                    input = dry.to(args.device)
                    target = wet.to(args.device)

                    pred = model(input, c)
                
                    # Pre-emphasis filter
                    pre_emphasis = config.get('pre_emphasis', None)
                    if pre_emphasis is not None:
                        pred = F.preemphasis(pred, float(pre_emphasis))
                    
                    loss1 = criterion1(pred, target)
                    
                    if config['criterion2'] is not None:
                        loss2 = criterion2(pred, target)
                    else:
                        loss2 = 0.0

                    loss = loss1 + loss2
                    
                    valid_loss += loss.item()                   
                avg_valid_loss = valid_loss / len(valid_loader)    
    
            writer.add_scalar('train/loss_valid', avg_valid_loss, global_step=current_epoch)
            
            scheduler.step(avg_valid_loss)

            # Save the model if it improved
            if avg_valid_loss < min_valid_loss:
                print(f"Epoch {epoch}: Loss improved from {min_valid_loss:4f} to {avg_valid_loss:4f} - > Saving model")
                min_valid_loss = avg_valid_loss
                save_model_checkpoint(
                    model, config, optimizer, scheduler, current_epoch, label, min_valid_loss, args
                )
            else:
                patience_count += 1
                # if config['early_stop_patience'] is not None and patience_count >= config['early_stop_patience']:
                if patience_count == config['early_stop_patience']:
                    print(f"Epoch {epoch}: Loss did not improve for {patience_count} epochs, stopping training")
                    break 
            
            current_epoch += 1

    except KeyboardInterrupt:
        print("\nTraining manually stopped by user. Processing the results...")

    finally:
        final_train_loss = float(avg_train_loss)
        final_valid_loss = float(avg_valid_loss)
        writer.add_hparams(config, {'train/final': final_train_loss, 'train/final_valid': final_valid_loss})
        print(f"Epoch {epoch}, final validation loss: {final_valid_loss}")

        if pre_emphasis is not None:
            pred = F.deemphasis(pred, float(pre_emphasis))

        pred = F.highpass_biquad(pred, config['sample_rate'], 10)
        target = F.highpass_biquad(target, config['sample_rate'], 10)

        pred = pred.view(-1).unsqueeze(0).cpu()
        target = target.view(-1).unsqueeze(0).cpu()

        target /= torch.max(torch.abs(target))                
        pred /= torch.max(torch.abs(pred))

        os.makedirs(f"{args.audio_dir}/train", exist_ok=True)

        save_out = f"{args.audio_dir}/train/pred-{label}.wav"
        torchaudio.save(save_out, pred, sample_rate=config['sample_rate'])

        save_target = f"{args.audio_dir}/train/targ-{label}.wav"
        torchaudio.save(save_target, target, sample_rate=config['sample_rate'])

        writer.flush()
        writer.close()

