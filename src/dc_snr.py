import torch
import csv

from src.egfxset import load_egfxset
from src.springset import load_springset
from configurations import parse_args


if __name__ == "__main__":

    args = parse_args()
    device = torch.device(args.device)
    sample_rate = args.sample_rate
    bit_rate = args.bit_rate


    def compute_dc_offset(signal):
        """
        Compute the DC offset of an audio signal.
        """
        return torch.mean(signal)

    def compute_snr(dry, wet):
        """
        Compute the Signal-to-Noise Ratio (SNR) using PyTorch.
        Values are in dB.
        """
        noise = dry - wet
        signal_power = torch.mean(dry**2)
        noise_power = torch.mean(noise**2)
        
        if noise_power == 0:  # Avoid log(0) which is undefined
            return float('inf')  # Infinite SNR
        
        snr_db = 10 * torch.log10(signal_power / noise_power)
        return snr_db.item()

    # Load data
    if args.dataset == 'egfxset':
        train_loader, _, _ = load_egfxset(args.datadir, batch_size=1, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    
    if args.dataset == 'springset':
        train_loader, _, test_loader = load_springset(args.datadir, batch_size=1, train_ratio=1.0)

    # Write to a CSV file
    with open(f'{args.dataset}_dc_esr.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Sample Index", "DC Offset", "SNR"])  # Writing headers

        for idx, (dry, wet) in enumerate(train_loader):
            # Move the tensors to the computation device
            dry = dry.to(device)
            wet = wet.to(device)
            
            # Compute metrics
            dc_offset_dry = compute_dc_offset(dry)
            dc_offset_wet = compute_dc_offset(wet)
            snr = compute_snr(dry, wet)

            writer.writerow([idx, dc_offset_dry.item(), dc_offset_wet.item(), snr])

            print(f"Sample {idx}: DC Offset = {dc_offset_dry:.3f}, {dc_offset_wet:.3f} SNR = {snr:.3f} dB", end='\r')

        print("Audio metrics saved to audio_metrics.csv!")
        