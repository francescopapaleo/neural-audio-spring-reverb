import torch

def modify_checkpoint_criteria(checkpoint_path, new_criterion):
    """
    Modify the 'criterion' field of a saved checkpoint.

    Args:
    - checkpoint_path (str): path to the checkpoint file.
    - new_criterion (str): new value for the 'criterion' field.
    """

    # 1. Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # 2. Modify the 'hparams' dictionary or any other desired field
    checkpoint['criterion'] = new_criterion

    # 3. Save the checkpoint back to disk
    torch.save(checkpoint, checkpoint_path)
    print(f"Criterion in {checkpoint_path} changed to {new_criterion}")

# Usage example:
checkpoint_path = "results/checkpoints/WaveNetFF/wavenet-ff-mse_47_4_20230823-022107.pt"
new_criterion = "mrstft"
modify_checkpoint_criteria(checkpoint_path, new_criterion)
