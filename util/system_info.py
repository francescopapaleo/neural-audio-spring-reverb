#!/homedtic/fpapaleo/.conda/envs jupyter

import sys
import platform
import pkg_resources
import torch

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

# Installed modules
# print("\nInstalled Modules:")
# installed_packages = pkg_resources.working_set
# for package in installed_packages:
#     print(f"{package.project_name} {package.version}")

# CUDA version
print("\nCUDA Version:")
print(torch.version.cuda)

if torch.cuda.is_available():
    device = "cuda"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")