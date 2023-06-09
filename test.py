""" Testing script
Test a trained model on the test set and save the results with Tensorboard.
"""

import torch
import torchaudio
import auraloss
from datetime import datetime
from data import SpringDataset
from argparse import ArgumentParser
from pathlib import Path

from TCN import TCNBase
from utils.plotter import plot_compare_waveform, plot_compare_spectrogram

def testing(load, data_dir, audio_dir, device, sample_rate):

    # set device                                                                                
    if device is None:                                                              
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load from checkpoint                                                                                                        
    try:
        checkpoint = torch.load(load, map_location=device)
        hparams = checkpoint['hparams']
    except Exception as e:
        print(f"Failed to load model state: {e}")
        return
    
    # instantiate model 
    model = TCNBase(                                                        
        n_inputs = hparams['n_inputs'], 
        n_outputs = hparams['n_outputs'], 
        n_blocks = hparams['n_blocks'],
        kernel_size = hparams['kernel_size'],
        n_channels = hparams['n_channels'], 
        dilation_growth = hparams['dilation_growth'],
        cond_dim = hparams['cond_dim'],
    ).to(device)

    # batch_size = hparams['batch_size']
    batch_size = hparams['batch_size']
    n_epochs = hparams['n_epochs']
    lr = hparams['l_rate']
    model_name = checkpoint['name']

    print(f"Loaded: {model_name}")

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Failed to load model state: {e}")
        return
    
    # load test dataset
    dataset = SpringDataset(root_dir=data_dir, split='test')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=0, drop_last=True)

    # initialize tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=f'runs/test01C/tcn_{n_epochs}_{batch_size}_{lr}_{timestamp}')
    

    print("## Initializing metrics...")
    l1 = torch.nn.L1Loss()
    esr = auraloss.time.ESRLoss()
    dc = auraloss.time.DCLoss()
    snr = auraloss.time.SNRLoss()
    
    criterions = [l1, esr, dc, snr]
    test_results = {"l1": [], "esr": [], "dc": [], "snr": []}

    # dummy condition tensor    
    c = torch.tensor([0.0, 0.0]).view(1,1,-1)           
    
    print("## Testing...")

    num_batches = len(test_loader)
    
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(test_loader):
            global_step = step + 1
            print(f"Batch {global_step}/{num_batches}")

            # move input and target to device
            input = input.to(device)                    
            target = target.to(device)
            c = c.to(device)

            # pad input and target to match receptive field
            rf = model.compute_receptive_field()
            input_pad = torch.nn.functional.pad(input, (rf-1, 0))
            target_pad = torch.nn.functional.pad(target, (rf-1, 0))
            
            # forward pass
            output = model(input_pad, c)
            output_trim = output[:,:,:target_pad.size(2)]

            # Compute metrics means for current batch
            for metric, name in zip(criterions, test_results.keys()):
                batch_score = metric(output_trim, target_pad).item()
                test_results[name].append(batch_score)
                
                # Write metrics to tensorboard
                writer.add_scalar(f'test/batch_{name}', batch_score, global_step)

             # Plot and save audios every 4 batches
            if step % 4 ==0:
                single_target = target_pad[0]
                single_output = output_trim[0]

                plot_compare_waveform(single_target.detach().cpu(), single_output.detach().cpu(),
                                        sample_rate, file_name=f"Waveform_{model_name}_{global_step}") 
                plot_compare_spectrogram(single_target.detach().cpu(), single_output.detach().cpu(), 
                                      sample_rate, file_name=f"Spectra_{model_name}_{global_step}", t_label=f"Target_{global_step}", o_label=f"Output_{global_step}",)
                
                audio_dir = Path(audio_dir)
                audio_dir.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(audio_dir / f"Target_{model_name}_{global_step}.wav"), single_target.detach().cpu(), sample_rate)
                torchaudio.save(str(audio_dir / f"Output_{model_name}_{global_step}.wav"), single_output.detach().cpu(), sample_rate)


    print("## Computing global metrics...")
    # compute global metrics means
    for name in test_results.keys():
        global_score = sum(test_results[name]) / len(test_results[name])
        writer.add_scalar(f'test/global_{name}', global_score, global_step)

    print("## Saving results...")
    # Flush and close the writer
    writer.flush()
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../plate-spring/spring/', help='Path (rel) to dataset ')
    parser.add_argument('--audio_dir', type=str, default='./audio/processed/', help='Path (rel) to audio files')
    parser.add_argument('--load', type=str, required=True, help='Path (rel) to checkpoint to load')
    parser.add_argument('--device', type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", help='set device to run the model on')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate of the audio')
    
    args = parser.parse_args()

    testing(args.load, args.data_dir, args.audio_dir, args.device, args.sample_rate)
    