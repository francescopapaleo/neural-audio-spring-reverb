import sys
import platform
import torch
import torchaudio

'''Print system information
.
├── README.md             # The top-level README for developers using this project
|
├── audio/                # Audio files
|  
├── data/
│
├── experiments/          # Experiment folders
│
├── models/               # Trained and serialized models
|
├── sandbox/              # For testing purposes
│
├── utils/               # Utility functions
│   ├── generator.py
│   ├── plot.py
│   └── system_info.py
│
├── config.py             # Configuration file
├── evaluation.py         # Evaluation script
├── main.py               # Main script
├── predict.py            # Prediction script
└── training.py           # Training script

'''

def system_info():
    # System information
    print("System Information:")
    print(f"System: {platform.system()}")
    print(f"Node: {platform.node()}")
    print(f"Release: {platform.release()}")
    print(f"Version: {platform.version()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    # Python version
    print("\nPython Version:")
    print(sys.version)

    # CUDA version
    print("\nCUDA Version:")
    print(torch.version.cuda)

    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")

    print(f'PyTorch version:{torch.__version__}')
    print(torchaudio.__version__)




if __name__ == "__main__":
    system_info()

