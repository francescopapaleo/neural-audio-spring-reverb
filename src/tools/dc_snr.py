import torch
import csv
from pathlib import Path

from src.default_args import parse_args
from src.data.egfxset import load_egfxset
from src.data.springset import load_springset

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


    def compute_snr(dry, wet, eps=1e-8):
        """
        Compute the Signal-to-Noise Ratio (SNR) using PyTorch.
        Values are in dB.
        """
        # reduce potential DC offset
        dry = dry - dry.mean()
        wet = wet - wet.mean()

        # compute SNR
        res = dry - wet
        snr = 10 * torch.log10(
            (wet ** 2).sum() / ((res ** 2).sum() + eps)
        )
        return snr.item()

    # Load data
    if args.dataset == 'egfxset':
        train_loader, _, _ = load_egfxset(args.datadir, batch_size=1, train_ratio=1.0, valid_ratio=0.0, test_ratio=0.0)
    
    elif args.dataset == 'springset':  # Added an 'elif' here for clarity.
        train_loader, _, _ = load_springset(args.datadir, batch_size=1, train_ratio=1.0)

    # Write to a CSV file
    destination = Path(args.logdir)/f'{args.dataset}_dc_snr.csv'

    with open(destination, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Sample Index", "dry_DC", "wet_DC", "SNR"])  # Fixed the headers 

        for idx, (dry, wet) in enumerate(train_loader):
            # Move the tensors to the computation device
            dry = dry.to(device)
            wet = wet.to(device)
            
            # Compute metrics
            dc_offset_dry = compute_dc_offset(dry)
            dc_offset_wet = compute_dc_offset(wet)
            snr = compute_snr(dry, wet)

            writer.writerow([idx, dc_offset_dry.item(), dc_offset_wet.item(), snr])

            print(f"Sample {idx}: DC Offset (dry) = {dc_offset_dry:.3f}, DC Offset (wet) = {dc_offset_wet:.3f}, SNR = {snr:.3f} dB", end='\r')

        print("\nAudio metrics saved to audio_metrics.csv!")  # Added a newline character for cleaner printing.
        