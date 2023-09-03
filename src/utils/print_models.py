import torch
from torchinfo import summary
from pathlib import Path
from pathlib import Path

from src.default_args import parse_args
from src.models.helpers import load_model_checkpoint, select_device


args = parse_args()

device = select_device(args.device)

# Path to the checkpoints
path_to_checkpoints = Path(args.models_dir)

# List all files in the directory
model_files = path_to_checkpoints.glob("*.pt")
with open(f'{args.results_dir}/model_summary.txt', 'w') as f:
    for model_file in model_files:

        # Load and print 
        f.write("\n")
        model, optimizer, scheduler, hparams, rf, params = load_model_checkpoint(device, model_file, args)

        f.write(f"Processing {model_file}...\n")
        f.write(f"Configuration name: {hparams['conf_name']}\n")
        f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        for key, value in hparams.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        if hparams['model_type'] in ["TCN", "PedalNetWaveNet", "GCN"]:
            rf = model.compute_receptive_field()
            f.write(f"Receptive field: {rf} samples or {(rf / args.sample_rate) * 1e3:0.1f} ms\n")
        else:
            rf = None
        
        # Print the model summary
        # inputs = torch.randn(4, 1, 240000).to(device)
        # model_summary = summary(model, input_data=inputs, verbose=0)
        # f.write(str(model_summary))  # convert the Summary object to a string
        # f.write("\n")

print("Done!")

