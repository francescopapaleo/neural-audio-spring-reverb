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
            "rtf"
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
        default="data/raw/",
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
        "--sample_rate",
        type=int,
        default=48000,
        help="The sample rate of the audio files (default: 48000)",
    )

    parser.add_argument(
        "--bit_depth",
        type=int,
        default=16,
        help="The bit depth of the audio files (default: 16)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to use for training (default: 32)",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="The number of workers to use for data loading (default: 4)",
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
        from .data.download import download_data

        download_data(args)
    elif args.action == "train":
        from .train import train_model

        train_model(args)
    elif args.action == "eval":
        from .eval import evaluate_model

        evaluate_model(args)
    elif args.action == "infer":
        from .inference import make_inference

        make_inference(args)
    elif args.action == "edit":
        from .utils.config_tools import modify_checkpoint

        modify_checkpoint(args)
    elif args.action == "report":
        from .utils.config_tools import print_models

        print_models(args)
    elif args.action == "ir":
        from .tools.ir_model import measure_model_ir

        measure_model_ir(args)
    elif args.action == "rt60":
        from .tools.rt60 import measure_rt60

        measure_rt60(args)
    elif args.action == "wrap":
        from .wrapper import wrap_model

        wrap_model(args)

    elif args.action == "rtf":
        from .rtf import measure_rtf

        measure_rtf(args)


if __name__ == "__main__":
    main()
