import os
import glob
from pathlib import Path

from src.default_args import parse_args
import src.inference as inf
import src.tools.ir_model as ir_model
import src.tools.rt60 as rt60
import src.test as test 
import src.train as train


def measure_ir(args):
    checkpoints_path = os.path.join(args.models_dir, '*.pt')
    for checkpoint in glob.glob(checkpoints_path):
        print(f"Measuring IRs: {checkpoint}")
        ir_model.measure(args)  # Assuming the function name is `measure`

def measure_rt60(args):
    IRs_path = os.path.join(args.audio_dir, 'measured_IR', '*.wav')
    for ir_audio in glob.glob(IRs_path):
        print(f"Measuring RT60: {ir_audio}")
        rt60.measure(args)  # Assuming the function name is `measure`

def test_checkpoints(args):
    for filename in os.listdir(args.models_dir):
        if filename.endswith(".pt"):
            full_path = Path(args.models_dir) / filename
            print(f"Testing checkpoint: {full_path}")
            args.checkpoint = full_path  # Assuming the test function uses 'args.checkpoint' to get the path
            test.test(args)  # Call the test function
            print("Done.\n")

def train_models(args):
    configs = ["tcn-baseline-v28", "wavenet-1k5-v28", "gcn-250-v28"]
    for config in configs:
        train(args)

def main():
    args = parse_args()

    test_checkpoints(args)
    
if __name__ == '__main__':
    main()