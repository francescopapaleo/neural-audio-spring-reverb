from torchinfo import summary
from src.models.helpers import load_model_checkpoint, select_device
from pathlib import Path
import torch
from src.default_args import parse_args

# Parse the arguments
args = parse_args()

# Select the device
device = select_device(args.device)

# Path to the checkpoints
path_to_checkpoints = Path(args.modelsdir)

# List all files in the directory
model_files = path_to_checkpoints.glob("*.pt")
with open('results/model_summary.txt', 'w') as f:
    for model_file in model_files:
        # Load model from the checkpoint
        model, model_name, hparams, optimizer_state_dict, scheduler_state_dict, last_epoch, rf, params = load_model_checkpoint(device, model_file, args)
        # Print information about the model
        f.write("\n")
        f.write(f"Processing {model_file}...")
        f.write(f"Configuration name: {hparams['conf_name']}\n")
        f.write(f"Model name: {model_name}\n")
        f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")
        f.write(f'criterion: {hparams["criterion"]}\n')
        f.write(f'batch size: {hparams["batch_size"]}\n')
        try:
            f.write(f'current epoch: {hparams["curr_epoch"]}\n')
            f.write(f'model path: {model_file}\n')
        except KeyError:
            pass

        if hparams['model_type'] in ["TCN", "WaveNet"]:
            rf = model.compute_receptive_field()
            f.write(f"Receptive field: {rf} samples or {(rf / args.sample_rate) * 1e3:0.1f} ms\n")
        else:
            rf = None
        
        # Print the model summary
        # inputs = torch.randn(4, 1, 240000).to(device)
        # model_summary = summary(model, input_data=inputs, verbose=0)
        # f.write(str(model_summary))  # convert the Summary object to a string
        f.write("\n")

print("Done!")

