import torch
import torchaudio
<<<<<<< HEAD
=======
import torchaudio.functional as F
>>>>>>> 48kHz
import auraloss
import numpy as np
import os

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path

from src.egfxset import load_egfxset
from src.springset import load_springset
from src.helpers import select_device, initialize_model, save_model_checkpoint, load_model_checkpoint
from configurations import parse_args, configs

<<<<<<< HEAD

def main():
    args = parse_args()

    torch.manual_seed(42)
    torch.cuda.empty_cache()            
    torch.backends.cudnn.benchmark = True

=======
def main():
    args = parse_args()
    torch.manual_seed(42)
    
>>>>>>> 48kHz
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
<<<<<<< HEAD
    writer.add_text('model_summary', str(model), global_step=0)

=======
>>>>>>> 48kHz
    torch.cuda.empty_cache()

    print(model)

    # Define loss function and optimizer
    mae = torch.nn.L1Loss().to(device)
    dc = auraloss.time.DCLoss().to(device)
    esr = auraloss.time.ESRLoss().to(device)
    mrstft =  auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048],
        win_lengths=[32, 128, 512, 2048],
<<<<<<< HEAD
        hop_sizes=[16, 64, 256, 1024]).to(device)
    dc = auraloss.time.DCLoss().to(device)

    criterion = mrstft
    criterion_str = 'mrstft'

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ms1 = int(args.n_epochs * 0.6)
    ms2 = int(args.n_epochs * 0.8)
    milestones = [ms1, ms2]
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1,verbose=False)
    
=======
        hop_sizes=[16, 64, 256, 1024],
        sample_rate=args.sample_rate,
        perceptual_weighting=True,
        ).to(device)
    
    criterion = hparams['criterion']
    alpha = 1.0

>>>>>>> 48kHz
    # Load data
    if args.dataset == 'egfxset':
        train_loader, valid_loader, _ = load_egfxset(args.datadir, args.batch_size)
    if args.dataset == 'springset':
        train_loader, valid_loader, _ = load_springset(args.datadir, args.batch_size)
    
    # Initialize minimum validation loss with infinity
    min_valid_loss = np.inf
<<<<<<< HEAD
    # patience_counter = 0
=======
>>>>>>> 48kHz

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
<<<<<<< HEAD
        for epoch in range(args.n_epochs):
=======
        for epoch in range(start_epoch, start_epoch + args.n_epochs):
>>>>>>> 48kHz
            train_loss = 0.0

            # Training
            model.train()
            c = torch.tensor([0.0, 0.0]).view(1,1,-1).to(device)
            for batch_idx, (dry, wet) in enumerate(train_loader):
                optimizer.zero_grad() 
<<<<<<< HEAD
                input = dry.to(device)
                target = wet.to(device)
            
                output = model(input, c)
                
                output = torchaudio.functional.preemphasis(output, 0.95)
                loss = criterion(output, target)        
                
=======
                input = dry.to(device)          # shape: [batch, channel, seq]
                target = wet.to(device)
                
                output = model(input)
                
                # output = torchaudio.functional.preemphasis(output, 0.95)
                loss_1 = mae(output, target)
                loss_2 = mrstft(output, target)
                loss = loss_1 + (alpha * loss_2)

>>>>>>> 48kHz
                loss.backward()                             
                optimizer.step()
         
                train_loss += loss.item()
                
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('training/learning_rate', lr, global_step=epoch)
<<<<<<< HEAD

            avg_train_loss = train_loss / len(train_loader)
            
            writer.add_scalar('training/loss', avg_train_loss, global_step=epoch)
            
            # Validation
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for step, (dry, wet) in enumerate(valid_loader):
                    input = dry.to(device)
                    target = wet.to(device)
                
                    output = model(input, c)
                    
                    output = torchaudio.functional.preemphasis(output, 0.95)
                    loss = criterion(output, target)
                    
                    valid_loss += loss.item()                   
                avg_valid_loss = valid_loss / len(valid_loader)    
    
            writer.add_scalar('validation/loss', avg_valid_loss, global_step=epoch)

            scheduler.step(avg_valid_loss) # Update learning rate
=======

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
>>>>>>> 48kHz

            # Save the model if it improved
            if avg_valid_loss < min_valid_loss:
                print(f"Epoch {epoch}: Loss improved from {min_valid_loss:4f} to {avg_valid_loss:4f} - > Saving model", end="\r")
<<<<<<< HEAD
                
                min_valid_loss = avg_valid_loss
                save_model_checkpoint(
                    model, hparams, criterion_str, optimizer, scheduler, args.n_epochs, args.batch_size, args.lr, timestamp, args
                )
                patience_counter = 0  # reset the counter if performance improved
            else:
                patience_counter += 1  # increase the counter if performance did not improve

            # # Early stopping if performance did not improve after n epochs
            # if patience_counter >= 20:
            #     print(f"Early stopping triggered after {patience_counter} epochs without improvement in validation loss.")
            #     break
=======
                min_valid_loss = avg_valid_loss
                save_model_checkpoint(
                    model, hparams, optimizer, scheduler, epoch, timestamp, avg_valid_loss, args
                )

>>>>>>> 48kHz
    finally:
        final_train_loss = avg_train_loss
        final_valid_loss = avg_valid_loss
        writer.add_hparams(hparams, {'Final Training Loss': final_train_loss, 'Final Validation Loss': final_valid_loss})
<<<<<<< HEAD
        print(f"Final Validation Loss: {final_valid_loss}")
        single_output = output[0].detach().cpu()
        abs_max = torch.max(torch.abs(single_output))
        # Avoid division by zero: if abs_max is 0, just set it to 1
        if abs_max == 0:
            abs_max = 1
        # Normalize waveform to range [-1, 1]
        normalized_output = single_output / abs_max

        # Save the normalized waveform
        save_path = f"{log_dir}/{args.config}_output.wav"
        torchaudio.save(save_path, normalized_output, args.sample_rate)
=======
        print(f"Final Validation Loss: {final_valid_loss}")    

        inp = input.view(-1).unsqueeze(0).cpu()
        tgt = target.view(-1).unsqueeze(0).cpu()
        out = output.view(-1).unsqueeze(0).cpu()

        inp /= torch.max(torch.abs(inp))
        tgt /= torch.max(torch.abs(tgt))                
        out /= torch.max(torch.abs(out))

        save_in = f"{log_dir}/inp_{hparams['conf_name']}.wav"
        torchaudio.save(save_in, inp, args.sample_rate)

        save_out = f"{log_dir}/out_{hparams['conf_name']}.wav"
        torchaudio.save(save_out, out, args.sample_rate)

        save_target = f"{log_dir}/tgt_{hparams['conf_name']}.wav"
        torchaudio.save(save_target, tgt, args.sample_rate)
>>>>>>> 48kHz

    writer.close()

if __name__ == "__main__":

    main()
