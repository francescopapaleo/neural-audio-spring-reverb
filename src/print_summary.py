from torchinfo import summary
from src.helpers import load_model_checkpoint, select_device
from pathlib import Path
import torch
from configurations import parse_args

# Parse the arguments
args = parse_args()

# Select the device
device = select_device(args.device)

# Path to the checkpoints
path_to_checkpoints = Path(args.checkpoint_path)

# List all files in the directory
model_files = path_to_checkpoints.glob("*.pt")
with open('model_summary.txt', 'w') as f:
        
    for model_file in model_files:
        # Load model from the checkpoint
        model, model_name, hparams = load_model_checkpoint(device, model_file)
        rf = hparams['receptive_field']

        # Print information about the model
        f.write("\n")
        f.write(f"Configuration name: {hparams['conf_name']}")
        f.write("\n")
        f.write(f"Model name: {model_name}")
        f.write("\n")
        if hparams['model_type'] in ["TCN", "WaveNet"]:
                rf = model.compute_receptive_field()
                f.write(f"Receptive field: {rf} samples or {(rf / args.sample_rate)*1e3:0.1f} ms")   
        else:
            rf = None
        f.write("\n")
        f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        f.write(f'hyperparameters: \n')
        f.write(str(hparams))
        f.write("\n")
        
        # Print the model summary
        inputs = torch.randn(8, 1, 32000).to(device)
        dummy_c = torch.randn(1, 1, 2).to(device)
        model_summary = summary(model, input_data=[inputs, dummy_c], verbose=0)
        f.write(str(model_summary))  # convert the Summary object to a string
        f.write("\n")


print("Done!")
