from config import parser
args = parser.parse_args()

import sys
import platform
import torch
import torchaudio

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

