import os
import torch
from argparse import ArgumentParser


def main():
    # Parse command line arguments
    parser = ArgumentParser(
        prog="python main.py [action] [options]",
        description="Neural Networks for spring reverb emulation",
        epilog="",
    )

    # Required arguments
    parser.add_argument(
        "action",
        choices=[
            "download",
            "train",
            "eval",
            "infer",
            "edit",
            "report",
            "ir",
            "rt60",
            "wrap",
        ],
        help="The action to perform, check the doc.",
    )

    # Optional arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda:0", "cpu"],
        help="Set device to run the model on (default: auto)",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Where the data will be downloaded to",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="audio",
        help="Relative path to the audio directory",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Relative path to the Tensorboard log directory",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="docs/assets/plots",
        help="Relative path to the plots directory",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="models",
        help="Relative path to the trained models directory",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Select the dataset to use (default: None)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="Relative path to the checkpoint file",
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="Relative path to the YAML file to initialize the model with",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Relative path to the input audio file",
    )

    parser.add_argument("--duration", type=float, default=5.0, help="")

    args = parser.parse_args()

    if args.device == "auto":
        # Automatically choose CUDA if available, else CPU
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(args.device)

    print(f"Using device: {args.device}")

    if args.action == "download":
        from src.data.download import download_data

        download_data(args)
    elif args.action == "train":
        from src.train import train_model

        train_model(args)
    elif args.action == "eval":
        from src.eval import evaluate_model

        evaluate_model(args)
    elif args.action == "infer":
        from src.inference import make_inference

        make_inference(args)
    elif args.action == "edit":
        from src.utils.config_tools import modify_checkpoint

        modify_checkpoint(args)
    elif args.action == "report":
        from src.utils.config_tools import print_models

        print_models(args)
    elif args.action == "ir":
        from src.tools.ir_model import measure_model_ir

        measure_model_ir(args)
    elif args.action == "rt60":
        from src.tools.rt60 import measure_rt60

        measure_rt60(args)
    elif args.action == "wrap":
        from src.wrapper import wrap_model

        wrap_model(args)


if __name__ == "__main__":
    main()
