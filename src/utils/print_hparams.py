import torch
import numpy as np

from src.models.helpers import load_model_checkpoint, save_model_checkpoint, select_device
from src.default_args import parse_args
from pathlib import Path

# Assuming your custom functions are in the same file, otherwise import them
# from your_custom_module import load_model_checkpoint, save_model_checkpoint

def modify_criterion_in_checkpoint(checkpoint_path, args):
    """
    Load the checkpoint, modify the hparams['criterion'] if needed, and save it back
    """
    device = select_device(args.device)

    # 1. Load the model checkpoint
    model, optimizer_state_dict, scheduler_state_dict, hparams, rf, params = load_model_checkpoint(device, checkpoint_path, args)

    # 2. Print the current value of hparams['criterion']
    print("Current hparams:")
    for key, value in hparams.items():
        print(f"{key}: {value}")

    # 3. Ask the user if they want to modify hparams['criterion']
    modify = input("Do you want to modify hparams['criterion']? (yes/no): ").strip().lower()

    if modify == "yes":
        # Get the new value
        new_value = input("Enter new value for hparams['criterion']: ").strip()

        # Modify the value in the hparams
        hparams['criterion'] = new_value
        print(f"Updated hparams['criterion'] to {new_value}")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
        optimizer.load_state_dict(optimizer_state_dict)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler.load_state_dict(scheduler_state_dict)
        
        # Save the modified checkpoint
        save_path = input("Enter path to save the modified checkpoint (leave empty to overwrite original): ").strip()
        if not save_path:
            save_path = checkpoint_path

        timestamp = "modified"  # Or any suitable string you prefer
        
        if 'curr_epoch' in hparams:
            hparams['state_epoch'] = hparams['curr_epoch']
            del hparams['curr_epoch']
        
        avg_valid_loss = hparams.get('avg_valid_loss', np.inf)
        if avg_valid_loss is None:
            avg_valid_loss = np.inf

        save_model_checkpoint(model, hparams, optimizer, scheduler, hparams['state_epoch'], timestamp, avg_valid_loss, args)
    

if __name__ == "__main__":

    args = parse_args()

    modify_criterion_in_checkpoint(args.checkpoint, args)
