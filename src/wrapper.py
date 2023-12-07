import os
import torch
import torch.nn as nn

from pathlib import Path
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter
from neutone_sdk.utils import save_neutone_model
from torch import Tensor
from typing import Dict, List
from src.networks.model_utils import load_model_checkpoint
from src.networks.custom_layers import Conv1dCausal


class PaddingCached(nn.Module):
    """Cached padding for cached convolutions."""

    def __init__(self, n_ch: int, padding: int) -> None:
        super().__init__()
        self.n_ch = n_ch
        self.padding = padding
        self.register_buffer("pad_buf", torch.zeros((1, n_ch, padding)))

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3  # (batch_size, in_ch, samples)
        bs = x.size(0)
        if bs > self.pad_buf.size(0):  # Perform resizing once if batch size is not 1
            self.pad_buf = self.pad_buf.repeat(bs, 1, 1)
        x = torch.cat([self.pad_buf, x], dim=-1)  # concat input signal to the cache
        self.pad_buf = x[..., -self.padding :]  # discard old cache
        return x


class Conv1dCached(nn.Module):  # Conv1d with cache
    """Cached causal convolution for streaming."""

    def __init__(self, convcausal: Conv1dCausal) -> None:
        super().__init__()
        padding = convcausal.padding  # input_len == output_len when stride=1
        self.pad = PaddingCached(convcausal.in_channels, convcausal.padding)
        self.conv = convcausal.conv

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)  # get (cached input + current input)
        x = self.conv(x)
        return x


def replace_modules(module):
    for name, child in module.named_children():
        if isinstance(child, Conv1dCausal):
            # Create a new instance of Conv1dCached using the Conv1dCausal instance
            cached_conv = Conv1dCached(child)
            # Replace the Conv1dCausal instance with the Conv1dCached instance
            setattr(module, name, cached_conv)
        else:
            # If the child is not a Conv1dCausal instance, call the function recursively
            replace_modules(child)


class GCNModelWrapper(WaveformToWaveformBase):
    def get_model_name(self) -> str:
        return "GCN.NeuralSpringReverb"  # <- EDIT THIS

    def get_model_authors(self) -> List[str]:
        return ["Francesco Papaleo"]  # <- EDIT THIS

    def get_model_short_description(self) -> str:
        return "Neural spring reverb effect"  # <- EDIT THIS

    def get_model_long_description(self) -> str:
        return """"""  # <- EDIT THIS

    def get_technical_description(self) -> str:
        return "GCN model based on the idea proposed by Comunità et al."  # <- EDIT THIS

    def get_tags(self) -> List[str]:
        return ["reverb"]  # <- EDIT THIS

    def get_model_version(self) -> str:
        return "1.0.0"  # <- EDIT THIS

    def is_experimental(self) -> bool:
        return True  # <- EDIT THIS

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Paper": "http://arxiv.org/abs/2211.00497.pdf",
            "Code": "https://github.com/mcomunita/gcn-tfilm",
        }  # <- EDIT THIS

    def get_citation(self) -> str:
        return """Comunità, M., Steinmetz, C. J., Phan, H., & Reiss, J. D. (2023). 
        Modelling Black-Box Audio Effects with Time-Varying Feature Modulation. 
        https://doi.org/10.1109/icassp49357.2023.10097173"""  # <- EDIT THIS

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("depth", "Modulation Depth", 0.5),
            NeutoneParameter("FiLM1", "Feature modulation 1", 0.0),
            NeutoneParameter("FiLM2", "Feature modulation 2", 0.0),
            NeutoneParameter("FiLM3", "Feature modulation 3", 0.0),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True  # Input is mono

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True  # Output is mono

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [48000]  # Set to model sample rate during training

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [1024]

    def do_forward_pass(self, x: Tensor, params: Dict[str, Tensor]) -> torch.Tensor:
        # conditioning for FiLM layer
        p1 = params["FiLM1"]
        p2 = params["FiLM2"]
        p3 = params["FiLM3"]
        depth = params["depth"]
        cond = torch.stack([p1, p2, p3], dim=1) * depth
        cond = cond.expand(x.shape[0], 3)

        # forward pass
        x = x.unsqueeze(1)
        x = self.model(x, cond)
        x = x.squeeze(1)
        return x


def wrap_model(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError("Checkpoint file not found")

    model, _, _, config, rf, params = load_model_checkpoint(args)
    model.eval()

    # Replace all instances of Conv1dCausal with Conv1dCached
    replace_modules(model)

    model_name = config["name"]
    destination_dir = Path(f"neutone_models/{model_name}")

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Export model to Neutone
    model = torch.jit.script(model.to("cpu"))
    model_wrapper = GCNModelWrapper(model)

    # Call the export function
    save_neutone_model(
        model=model_wrapper,
        root_dir=destination_dir,
        dump_samples=True,
        submission=True,
        max_n_samples=3,
        freeze=False,
        optimize=False,
        speed_benchmark=True,
    )
