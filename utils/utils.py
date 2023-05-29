# Utility functions
import torch

def center_crop(x, length: int):
    start = (x.shape[-1]-length)//2
    stop  = start + length
    return x[...,start:stop]

@torch.jit.unused
def causal_crop(x, length: int):
    if x.shape[-1] != length:
        stop = x.shape[-1] - 1
        start = stop - length
        x = x[..., start:stop]
    return x

