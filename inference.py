# inference.py: Use a pre-trained model to make inference on a given audio file

from pathlib import Path
import torch
import torchaudio
from argparse import ArgumentParser
import scipy.signal

from models.TCN import TCNBase

def make_inference(load: str, 
                   input_path: str, 
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
    x_p, fs_x = torchaudio.load(input_path)
    x_p = x_p.float()
    # calculate max_length if not provided
    if max_length is None:
        max_length = x_p.shape[1] / fs_x

    # Receptive field
    rf = model.compute_receptive_field()
    
    # Crop input signal if needed
    max_samples = int(fs_x * max_length)
    x_p_crop = x_p[:, :max_samples]
    chs = x_p_crop.shape[0]

    # If mono and stereo requested
    if chs == 1 and stereo:
        x_p_crop = x_p_crop.repeat(2, 1)
        chs = 2

    # Pad the input signal
    front_pad = rf - 1
    back_pad = 0 if not tail else front_pad
    x_p_pad = torch.nn.functional.pad(x_p_crop, (front_pad, back_pad))

    # Design highpass filter
    sos = scipy.signal.butter(
        8,
        20.0,
        fs=fs_x,
        output="sos",
        btype="highpass"
    )

    # Compute linear gain
    gain_ln = 10 ** (gain_dB / 20.0)

    # Process audio with the pre-trained model
    with torch.no_grad():
        y_hat = torch.zeros(x_p_crop.shape[0], max_samples + back_pad)      # was: y_hat = torch.zeros(x_p_crop.shape[0], x_p_crop.shape[1] + back_pad)

        for n in range(chs):
            if n == 0:
                factor = (width * 5e-3)
            elif n == 1:
                factor = -(width * 5e-3)
            c = torch.tensor([float(c0 + factor), float(c1 + factor)]).view(1, 1, -1)
            
            print("x_p_pad[n, :].shape:", x_p_pad[n, :].shape) 
            y_hat_ch = model(gain_ln * x_p_pad[n, :].view(1, 1, -1), c)

            print("After model y_hat_ch.shape:", y_hat_ch.shape)
            y_hat_ch = scipy.signal.sosfilt(sos, y_hat_ch.view(-1).numpy())

            print("After sosfilt y_hat_ch.shape:", y_hat_ch.shape)
            y_hat_ch = torch.tensor(y_hat_ch)
            # compute the padding needed
            padding_needed = max_samples + back_pad - y_hat_ch.size(0)

            # Pad the tensor
            if padding_needed > 0:
                y_hat_ch = torch.nn.functional.pad(y_hat_ch, (0, padding_needed))
            
            y_hat[n, :] = y_hat_ch

    # Pad the dry signal
    x_dry = torch.nn.functional.pad(x_p_crop, (0, back_pad))

    # Normalize each first
    y_hat /= y_hat.abs().max()
    x_dry /= x_dry.abs().max()

    # Mix
    mix = mix / 100.0
    x_dry_padded = torch.nn.functional.pad(x_dry, (0, y_hat.size(1) - x_dry.size(1)))   # ADDED

    y_hat = (mix * y_hat) + ((1 - mix) * x_dry_padded)

    # Remove transient
    y_hat = y_hat[..., 8192:]
    y_hat /= y_hat.abs().max()

    # Save the output using torchaudio
    input_file_name = Path(input_path).stem
    output_file_name = Path('./data/processed') / (input_file_name + "_processed.wav")
    torchaudio.save(str(output_file_name), y_hat, sample_rate=sample_rate, channels_first=True, bits_per_sample=16)
    print(f"Saved processed file to {output_file_name}")

    return y_hat

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--load', type=str, default=None, help='Path to the checkpoint to load')

    # Specific inference arguments
    parser.add_argument('--input', type=str, required=True, help='path to input audio file')
    parser.add_argument('--max_length', type=float, default=None, help='maximum length of the output audio')
    parser.add_argument('--stereo', type=bool, default=False, help='flag to indicate if the audio is stereo or mono')
    parser.add_argument('--tail', type=bool, default=None, help='flag to indicate if tail padding is required')
    parser.add_argument('--width', type=float, default=50, help='width parameter for the model')
    parser.add_argument('--c0', type=float, default=0, help='c0 parameter for the model')
    parser.add_argument('--c1', type=float, default=0, help='c1 parameter for the model')
    parser.add_argument('--gain_dB', type=float, default=0, help='gain in dB for the model')
    parser.add_argument('--mix', type=float, default=50, help='mix parameter for the model')
    args = parser.parse_args()

    make_inference(args.input, args.sample_rate, args.device, args.load, args.max_length, args.stereo, args.tail, args.width, args.c0, args.c1, args.gain_dB, args.mix)
