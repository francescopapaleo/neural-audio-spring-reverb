import torch
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
    model, model_name, hparams, conf_settings, optimizer_state_dict, scheduler_state_dict, rf, params = load_model_checkpoint(device, checkpoint_path, args)

    # 2. Print the current value of hparams['criterion']
    print("Current hparams:")
    print(hparams)

    print("Current conf_settings:")
    print(hparams)

    # 3. Ask the user if they want to modify hparams['criterion']
    modify = input("Do you want to modify hparams['criterion']? (yes/no): ").strip().lower()

    if modify == "yes":
        # Get the new value
        new_value = input("Enter new value for hparams['criterion']: ").strip()

        # Modify the value in the hparams
        hparams['criterion'] = new_value
        print(f"Updated hparams['criterion'] to {new_value}")

        # Create dummy optimizer and scheduler as placeholders since your save function expects them
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Just a placeholder
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Just a placeholder

        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

        # Save the modified checkpoint
        save_path = input("Enter path to save the modified checkpoint (leave empty to overwrite original): ").strip()
        if not save_path:
            save_path = checkpoint_path

        # Assuming you're okay with using dummy values for these
        epoch = conf_settings['state_epoch']
        timestamp = "modified"  # Or any suitable string you prefer
        avg_valid_loss = conf_settings.get('avg_valid_loss', 0.0)

        save_model_checkpoint(model, hparams, conf_settings, optimizer, scheduler, epoch, timestamp, avg_valid_loss)
    

if __name__ == "__main__":

    args = parse_args()


    # 2. Modify the criterion in the checkpoint if needed
    modify_criterion_in_checkpoint(args.checkpoint, args)
