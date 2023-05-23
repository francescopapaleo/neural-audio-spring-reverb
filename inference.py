import os
from pathlib import Path

import torch
import torchaudio
import torchsummary
import matplotlib
from matplotlib import pyplot as plt

from utils.rt60_compute import measure_rt60
import numpy as np
import scipy.signal

from tcn import TCN, causal_crop, model_params
from config import parser

torch.backends.cudnn.benchmark = True

def make_inference():
    args = parser.parse_args()

    models_dir = args.models_dir
    model_file = args.load
    load_from = Path(models_dir) / model_file
    
    print("loading selected model")
    model = TCN(
        n_inputs=1,
        n_outputs=1,
        cond_dim=model_params["cond_dim"], 
        kernel_size=model_params["kernel_size"], 
        n_blocks=model_params["n_blocks"], 
        dilation_growth=model_params["dilation_growth"], 
        n_channels=model_params["n_channels"],
    )

    model.load_state_dict(torch.load(load_from))
    model = model.to(parser.parse_args().device)  # move the model to the right device
    model.eval()  # set the model to evaluation mode

    torchsummary.summary(model, [(1, 65536), (1, 2)], device=args.device)

    # Define the additional processing parameters
    gain_dB = model_params["gain_dB"]
    c0 = model_params["c0"]
    c1 = model_params["c1"]
    mix = model_params["mix"]
    width = model_params["width"]
    max_length = model_params["max_length"]
    stereo = model_params["stereo"]
    tail = model_params["tail"]


    input_path = Path(args.audio_dir) / args.input
    x_p, sample_rate = torchaudio.load(input_path)
    x_p = x_p.float()

    # Receptive field
    pt_model_rf = model.compute_receptive_field()

    # Crop input signal if needed
    max_samples = int(sample_rate * max_length)
    x_p_crop = x_p[:, :max_samples]
    chs = x_p_crop.shape[0]

    # If mono and stereo requested
    if chs == 1 and stereo:
        x_p_crop = x_p_crop.repeat(2, 1)
        chs = 2

    # Pad the input signal
    front_pad = pt_model_rf - 1
    back_pad = 0 if not tail else front_pad
    x_p_pad = torch.nn.functional.pad(x_p_crop, (front_pad, back_pad))

    # Design highpass filter
    sos = scipy.signal.butter(
        8,
        20.0,
        fs=sample_rate,
        output="sos",
        btype="highpass"
    )

    # Compute linear gain
    gain_ln = 10 ** (gain_dB / 20.0)

    # Process audio with the pre-trained model
    with torch.no_grad():
        y_hat = torch.zeros(x_p_crop.shape[0], x_p_crop.shape[1] + back_pad)
        for n in range(chs):
            if n == 0:
                factor = (width * 5e-3)
            elif n == 1:
                factor = -(width * 5e-3)
            c = torch.tensor([float(c0 + factor), float(c1 + factor)]).view(1, 1, -1)
            y_hat_ch = model(gain_ln * x_p_pad[n, :].view(1, 1, -1), c)
            y_hat_ch = scipy.signal.sosfilt(sos, y_hat_ch.view(-1).numpy())
            y_hat_ch = torch.tensor(y_hat_ch)
            y_hat[n, :] = y_hat_ch

    # Pad the dry signal
    x_dry = torch.nn.functional.pad(x_p_crop, (0, back_pad))

    # Normalize each first
    y_hat /= y_hat.abs().max()
    x_dry /= x_dry.abs().max()

    # Mix
    mix = mix / 100.0
    y_hat = (mix * y_hat) + ((1 - mix) * x_dry)

    # Remove transient
    y_hat = y_hat[..., 8192:]
    y_hat /= y_hat.abs().max()

    # Save the output using torchaudio
    output_file_name = Path(args.audio_dir).stem + "_processed.wav"
    torchaudio.save(str(output_file_name), y_hat, sample_rate=int(sample_rate))

    # Measure RT60 of the output signal
    if rt60:
        rt60 = measure_rt60(y_hat[0].numpy(), fs=sample_rate, plot=True, rt60_tgt=4.0)
        print("Estimated RT60:", rt60)

    return y_hat

if __name__ == "__main__":
    
    make_inference()
