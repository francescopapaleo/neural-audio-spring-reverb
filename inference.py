import argparse
from config import *

import torch
import torchaudio
import scipy.signal
import scipy.fftpack
import numpy as np

from pathlib import Path

from model import TCN, causal_crop
from utils.plot import plot_compare_waveform, plot_zoom_waveform, get_spectrogram, plot_compare_spectrogram, plot_transfer_function
from utils.rt60_measure import measure_rt60
from utils.generator import generate_reference

def make_inference(input_file, rt60=True):

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the model
    model = TCN(
        n_inputs=INPUT_CH,
        n_outputs=OUTPUT_CH,
        cond_dim=model_params["cond_dim"], 
        kernel_size=model_params["kernel_size"], 
        n_blocks=model_params["n_blocks"], 
        dilation_growth=model_params["dilation_growth"], 
        n_channels=model_params["n_channels"])

    # Load the state dictionary
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    model.eval()

    # Define the additional processing parameters
    gain_dB = processing_params["gain_dB"]
    c0 = processing_params["c0"]
    c1 = processing_params["c1"]
    mix = processing_params["mix"]
    width = processing_params["width"]
    max_length = processing_params["max_length"]
    stereo = processing_params["stereo"]
    tail = processing_params["tail"]
    output_file = processing_params["output_file"]
  
    if isinstance(input_file, str):
        # if the input is a string, we assume it's a file path
        input_file_path = input_file
        x_p, sample_rate = torchaudio.load(input_file_path)
        x_p = x_p.float()

    # if the input is a numpy array, convert it to a tensor
    elif isinstance(input_file, np.ndarray):
        x_p = torch.from_numpy(input_file).unsqueeze(0)
        sample_rate = SAMPLE_RATE
        x_p = x_p.float()
    else:
        raise ValueError("Invalid input type. Expected file path or numpy array.")


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
    if isinstance(input_file, np.ndarray):
        output_file_name = "processed.wav"  # Or whatever makes sense in your application
    else:
        output_file_name = Path(input_file).stem + "_processed.wav"

    # Measure RT60 of the output signal
    if rt60:
        rt60 = measure_rt60(y_hat[0].numpy(), fs=sample_rate, plot=True, rt60_tgt=4.0)
        print("Estimated RT60:", rt60)

    return y_hat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the inference script.")
    parser.add_argument("--input_file", help="Path to the input file.")
    args = parser.parse_args()

    make_inference(args.input_file)