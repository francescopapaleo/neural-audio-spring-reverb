""" Inference
Use a pre-trained model to make inference on a given audio file or numpy array.
"""

from pathlib import Path
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from argparse import ArgumentParser

from TCN import TCNBase


def load_audio(input, sample_rate):
    print(f"Input type: {type(input)}")  # add this line to check the type of the input
    if isinstance(input, str):
        # Load audio file
        x_p, fs_x = torchaudio.load(input)
        x_p = x_p.float()
        input_name = Path(input).stem
    elif isinstance(input, np.ndarray):  # <-- change here
        # Convert numpy array to tensor and ensure it's float32
        x_p = torch.from_numpy(input).float()
        # Add an extra dimension if necessary to simulate channel dimension
        if len(x_p.shape) == 1:
            x_p = x_p.unsqueeze(0)
        fs_x = sample_rate
        input_name = 'sweep'
    else:
        raise ValueError('input must be either a file path or a numpy array')
    
    return x_p, fs_x, input_name



def make_inference(load: str, 
                   input: any, 
                   sample_rate: int, 
                   device: str,
                   max_length: float, 
                   stereo: bool, 
                   tail: float, 
                   width: float, 
                   c0: float,
                   c1: float,
                   gain_dB: float,
                   mix: float
                   ) -> torch.Tensor:
    """
    Make inference on a given audio file using a pre-trained model.

    Parameters
    ----------
    load : str
        Path to the checkpoint file to load.
    input_path : str
        Path to the input audio file.
    sample_rate : int
        Sample rate of the input audio file.
    device : str
        Device to use for inference (usually 'cpu' or 'cuda').
    max_length : float
        Maximum length of the input signal in seconds.
    stereo : bool
        Whether to process input as stereo audio. If true and input is mono, it will be duplicated to two channels.
    tail : float
        Length of reverb tail in seconds.
    width : float
        Width of stereo field (only applicable for stereo input).
    c0 : float
        Conditioning parameter defined by the user.
    c1 : float
        Conditioning parameter defined by the user.
    gain_dB : float
        Gain of the output signal in dB.
    mix : float
        Proportion of dry/wet signal in the output (expressed as a percentage).

    Returns
    -------
    torch.Tensor
        Output audio after processing.
    """

    # set device                                                                                
    if device is None:                                                              
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load from checkpoint                                                                                                     
    try:
        checkpoint = torch.load(load, map_location=device)
        hparams = checkpoint['hparams']
    except Exception as e:
        print(f"Failed to load model state: {e}")
        return
    
    # instantiate model 
    model = TCNBase(                                                        
        n_inputs = hparams['n_inputs'], 
        n_outputs = hparams['n_outputs'], 
        n_blocks = hparams['n_blocks'],
        kernel_size = hparams['kernel_size'],
        n_channels = hparams['n_channels'], 
        dilation_growth = hparams['dilation_growth'],
        cond_dim = hparams['cond_dim'],
    ).to(device)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Failed to load model state: {e}")
        return

    # Load the audio
    x_p, fs_x, input_name = load_audio(input, sample_rate)
    
    # Receptive field
    rf = model.compute_receptive_field()
    
    chs = x_p.shape[0]

    # If mono and stereo requested
    if chs == 1 and stereo:
        x_p = x_p.repeat(2, 1)
        chs = 2

    # Pad the input signal 
    front_pad = rf - 1
    back_pad = 0 if not tail else front_pad
    x_p_pad = torch.nn.functional.pad(x_p, (front_pad, back_pad))

    # Compute linear gain
    gain_ln = 10 ** (gain_dB / 20.0)

    # Process audio with the pre-trained model
    with torch.no_grad():
        y_wet = torch.zeros((chs, x_p_pad.shape[1]))

        for n in range(chs):
            if n == 0:
                factor = (width * 5e-3)
            elif n == 1:
                factor = -(width * 5e-3)
            c = torch.tensor([float(c0 + factor), float(c1 + factor)]).view(1, 1, -1)
        
            y_wet_ch = model(gain_ln * x_p_pad[n, :].view(1, 1, -1), c)

            y_wet_ch = F.highpass_biquad(y_wet_ch.view(-1), fs_x, 20.0)
            y_wet_ch = F.lowpass_biquad(y_wet_ch.view(-1), fs_x, 20000.0)

            y_wet[n, :] = y_wet_ch

    x_dry = x_p_pad

    # Normalize each first
    y_wet /= y_wet.abs().max()
    x_dry /= x_p_pad.abs().max()

    # Mix
    mix = mix / 100.0
    y_hat = (mix * y_wet) + ((1 - mix) * x_dry)

    # # Remove transient
    y_hat = y_hat[..., 8192:]
    y_hat /= y_hat.abs().max()

    # Save the output using torchaudio
    output_file_name = Path('./data/processed') / (input_name + '_' + load + '.wav')
    torchaudio.save(str(output_file_name), y_hat, sample_rate=sample_rate, channels_first=True, bits_per_sample=16)
    print(f"Saved processed file to {output_file_name}")

    return y_hat


if __name__ == "__main__":
    

    parser = ArgumentParser()

    parser.add_argument('--load', type=str, default=None, help='relative path to checkpoint to load')
    parser.add_argument('--input', type=str, default=None, help='relative path to input audio to load')
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--sample_rate', type=int, default=16000)
    
    parser.add_argument('--max_length', type=float, default=None, help='maximum length of the output audio')
    parser.add_argument('--stereo', type=bool, default=False, help='flag to indicate if the audio is stereo or mono')
    parser.add_argument('--tail', type=bool, default=None, help='flag to indicate if tail padding is required')
    parser.add_argument('--width', type=float, default=0, help='width parameter for the model')
    parser.add_argument('--c0', type=float, default=0, help='c0 parameter for the model')
    parser.add_argument('--c1', type=float, default=0, help='c1 parameter for the model')
    parser.add_argument('--gain_dB', type=float, default=0, help='gain in dB for the model')
    parser.add_argument('--mix', type=float, default=100, help='mix parameter for the model')
    args = parser.parse_args()

    make_inference(args.load, args.input, args.sample_rate, args.device, args.max_length, args.stereo, args.tail, args.width, args.c0, args.c1, args.gain_dB, args.mix)

