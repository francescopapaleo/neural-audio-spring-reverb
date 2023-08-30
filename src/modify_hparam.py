import torch
from configurations import parse_args

args = parse_args()

def modify_checkpoint_criteria(device, checkpoint_path, new_variable, args):
    """
    Modify the 'criterion' field of a saved checkpoint.

    Args:
    - checkpoint_path (str): path to the checkpoint file.
    - new_criterion (str): new value for the 'criterion' field.
    """

    # 1. Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. Modify the 'hparams' dictionary or any other desired field
    checkpoint['name'] = new_variable

    # 3. Save the checkpoint back to disk
    torch.save(checkpoint, checkpoint_path)
    print(f"Hparam in {checkpoint_path} changed {new_variable}")

# Usage example:
checkpoint_path = "results/48k/models/tcn-baseline-48.0k.pt"
new_variable = "tcn-5500"
modify_checkpoint_criteria(args.device, checkpoint_path, new_variable, args)
