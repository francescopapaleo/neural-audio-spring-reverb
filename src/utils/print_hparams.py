import torch
from src.default_args import parse_args

def print_hparam(device, checkpoint_path, args):
    """
    Print the hparams of a checkpoint
    """

    # 1. Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 2. Print the hparams
    print(f"Content: {checkpoint.keys()}")
    print("")
    print(checkpoint['hparams'])


if __name__ == "__main__":

    args = parse_args()

    # 1. Select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 2. Print the hparams
    print_hparam(device, args.checkpoint, args)
