import torch
import numpy as np
from pathlib import Path
import traceback

from src.networks.checkpoints import load_model_checkpoint, save_model_checkpoint


def modify_checkpoint(args):
    """
    Load the checkpoint, modify the config['criterion'] if needed, and save it back

    Args:
    """
    # 1. Load the checkpoint and print config
    (
        model,
        optimizer_state_dict,
        scheduler_state_dict,
        config,
        rf,
        params,
    ) = load_model_checkpoint(args)
    print("|------------------------------------------------------|")
    print("Current config:")
    for key, value in config.items():
        print(f"{key}: {value}")
    print("|------------------------------------------------------|")

    # 2. Ask the user if they want to modify config
    key_to_modify = input(
        "Enter the hparam you want to modify (e.g., 'criterion', 'lr', hit ENTER to abort: "
    ).strip()
    if not key_to_modify:
        print("Exiting without making any modifications.")
        return

    # 3. Check if the entered hparam exists and prompt the user for the new value
    if key_to_modify in config:
        new_value = input(f"Enter new value for config['{key_to_modify}']: ").strip()

        # 4. Check the type of the existing value and cast the new value to the same type
        if isinstance(config[key_to_modify], float):
            try:
                new_value = float(new_value)
            except ValueError:
                print(f"Expected a float value for '{key_to_modify}'. Update aborted.")
                return
        elif isinstance(config[key_to_modify], int):
            try:
                new_value = int(new_value)
            except ValueError:
                print(
                    f"Expected an integer value for '{key_to_modify}'. Update aborted."
                )
                return
        elif isinstance(config[key_to_modify], str):
            pass
        else:
            print(f"Unhandled data type for '{key_to_modify}'. Update aborted.")
            return

    # Adding 'pre_emphasis' key
    elif key_to_modify == "pre_emphasis":
        new_value = input(f"Enter new value for config['{key_to_modify}']: ").strip()
        try:
            new_value = float(new_value) or None

        except ValueError:
            print(
                f"Expected a float or None value for '{key_to_modify}'. Update aborted."
            )
            return
        pass
    else:
        print(f"'{key_to_modify}' is not a recognized hyperparameter.")
        return

    # 5. Update the config
    config[key_to_modify] = new_value
    print(f"Updated config['{key_to_modify}'] to {new_value}")

    # 6. Load the optimizer and scheduler state dicts to save the model as it was
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    scheduler.load_state_dict(scheduler_state_dict)

    timestamp = "modified"

    save_model_checkpoint(
        model, config, optimizer, scheduler, 56, timestamp, np.inf, args
    )


def print_models(args):
    """
    This function prints the configuration dict and the model summary for all the models in the models directory.
    """
    # Path to the checkpoints
    path_to_checkpoints = Path(args.models_dir)

    # List all files in the directory
    model_files = path_to_checkpoints.glob("*.pt")
    with open(f"{args.log_dir}/model_summary.txt", "w") as f:
        for model_file in model_files:
            try:
                args.checkpoint = model_file
                f.write(
                    "|------------------------------------------------------------------------------------------|\n"
                )
                print(f"Current File: {Path(model_file)}")
                # Load and print
                f.write("\n")
                model, optimizer, scheduler, config, rf, params = load_model_checkpoint(
                    args
                )
                f.write(f"Processing {Path(model_file)}...\n")
                f.write(f"Configuration name: {config['name']}\n")
                f.write(f"Model type: {config['model_type']}\n")
                print(f"Model type: {config['model_type']}")
                f.write(
                    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n"
                )
                for key, value in config.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                if config["model_type"] in ["TCN", "WaveNet", "GCN_FiLM"]:
                    rf = model.compute_receptive_field()
                    f.write(
                        f"Receptive field: {rf} samples or {(rf / config['sample_rate']) * 1e3:0.1f} ms\n"
                    )
                else:
                    rf = None
                f.write("\n")
                f.write(f"Model architecture:\n")
                # convert the model to a string and write it to the file
                f.write(str(model))
                f.write("\n")
                print("\n\n")

            except Exception as e:
                # Handle the exception (e.g., print an error message)
                f.write(f"Error loading {Path(model_file)}:\n")
                f.write(str(e))
                f.write("\n")
                traceback.print_exc()  # Print the traceback for debugging
                continue  # Continue processing other models

    print("Done!")
