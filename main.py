import os
import glob
from pathlib import Path

from src.default_args import parse_args
from src.test import test_model

ROOT_DIR = Path(__file__).parent



# def measure_ir(args):
#     checkpoints_path = os.path.join(args.models_dir, '*.pt')
#     for checkpoint in glob.glob(checkpoints_path):
#         print(f"Measuring IRs: {checkpoint}")
#         ir_model.measure(args)  # Assuming the function name is `measure`

# def measure_rt60(args):
#     IRs_path = os.path.join(args.audio_dir, 'measured_IR', '*.wav')
#     for ir_audio in glob.glob(IRs_path):
#         print(f"Measuring RT60: {ir_audio}")
#         rt60.measure(args)  # Assuming the function name is `measure`

def test_checkpoints(args):
    path_to_checkpoints = Path(ROOT_DIR) / args.models_dir
    model_files = path_to_checkpoints.glob("*.pt")
    for model_file in model_files:
        args.checkpoint = model_file  # Assuming the test function uses 'args.checkpoint' to get the path
        print(f"Testing checkpoint: {args.checkpoint}")
        
        test_model(args)  # Call the test function
        print("Done.\n")

# def train_models(args):
#     configs = ["tcn-baseline-v28", "wavenet-1k5-v28", "gcn-250-v28"]
#     for config in configs:
#         args.conf = config
#         train_model(args)

def main():
    args = parse_args()

    test_checkpoints(args)
    
if __name__ == '__main__':
    main()