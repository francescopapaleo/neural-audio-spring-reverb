import torch
import numpy as np
from pathlib import Path

from src.default_args import parse_args
from src.models.helpers import load_model_checkpoint, save_model_checkpoint, select_device


def modify_criterion_in_checkpoint(checkpoint_path, args):
    """
    Load the checkpoint, modify the hparams['criterion'] if needed, and save it back

    Parameters
    """
    device = select_device(args.device)

    # 1. Load the checkpoint and print hparams
    model, optimizer_state_dict, scheduler_state_dict, hparams, rf, params = load_model_checkpoint(device, checkpoint_path, args)
    print("Current hparams:")
    for key, value in hparams.items():
        print(f"{key}: {value}")

    # 2. Ask the user if they want to modify hparams
    key_to_modify = input("Enter the hparam you want to modify (e.g., 'criterion', 'lr', hit ENTER to abort: ").strip()
    if not key_to_modify:
        print("Exiting without making any modifications.")
        return

    # 3. Check if the entered hparam exists and prompt the user for the new value
    if key_to_modify in hparams:
        new_value = input(f"Enter new value for hparams['{key_to_modify}']: ").strip()

        # 4. Check the type of the existing value and cast the new value to the same type
        if isinstance(hparams[key_to_modify], float):
            try:
                new_value = float(new_value)
            except ValueError:
                print(f"Expected a float value for '{key_to_modify}'. Update aborted.")
                return
        elif isinstance(hparams[key_to_modify], int):
            try:
                new_value = int(new_value)
            except ValueError:
                print(f"Expected an integer value for '{key_to_modify}'. Update aborted.")
                return
        elif isinstance(hparams[key_to_modify], str):
            pass
        else:
            print(f"Unhandled data type for '{key_to_modify}'. Update aborted.")
            return
    # Adding 'pre_emphasis' key
    elif key_to_modify == 'pre_emphasis':
        new_value = input(f"Enter new value for hparams['{key_to_modify}']: ").strip()
        try:
            new_value = float(new_value) or None

        except ValueError:
            print(f"Expected a float or None value for '{key_to_modify}'. Update aborted.")
            return
        pass
    else:
        print(f"'{key_to_modify}' is not a recognized hyperparameter.")
        return 

    # 5. Update the hparams
    hparams[key_to_modify] = new_value
    print(f"Updated hparams['{key_to_modify}'] to {new_value}")

    # 6. Load the optimizer and scheduler state dicts to save the model as it was
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])
    optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    scheduler.load_state_dict(scheduler_state_dict)
        
    # 7. Save the modified checkpoint
    save_path = input("Enter path to save the modified checkpoint (leave empty to overwrite original): ").strip()
    if not save_path:
        save_path = checkpoint_path

    timestamp = "modified"
    
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
