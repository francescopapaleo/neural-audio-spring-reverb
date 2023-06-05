# inference.py: The code for the inference script

from pathlib import Path
import torch
import torchaudio
from argparse import ArgumentParser
import auraloss
import scipy.signal

from utils.rt60 import measure_rt60
from utils.tf import compute_tf 
from models.TCN import TCNBase, causal_crop

def make_inference(args):

    # parse default arguments
    data_dir = args.data_dir
    device = args.device
    sample_rate = args.sample_rate
    lr = args.lr

    # set device                                                                                
    if device is None:                                                              
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # load from checkpoint                                                                            
    load_from = Path('./checkpoints') / (args.load + '.pt')                                   
    try:
        checkpoint = torch.load(load_from, map_location=device)
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

    # inference-specific arguments
    input_path = args.input_path
    max_length = args.max_length
    stereo = args.stereo
    tail = args.tail
    width = args.width
    c0 = args.c0
    c1 = args.c1
    gain_dB = args.gain_dB
    mix = args.mix

    # Load the audio
    x_p, fs_x = torchaudio.load(input_path)
    x_p = x_p.float()

    # Receptive field
    pt_model_rf = model.compute_receptive_field()
    
    # Crop input signal if needed
    max_samples = int(fs_x * max_length)
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
        fs=fs_x,
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
    output_file_name = Path(args.results_dir).stem + "_processed.wav"
    torchaudio.save(str(output_file_name), y_hat, sample_rate=int(fs_x))

    # Measure RT60 of the output signal
    rt60 = measure_rt60(y_hat[0].numpy(), sample_rate=fs_x, plot=True, rt60_tgt=4.0)
    print("Estimated RT60:", rt60)

    return y_hat

if __name__ == "__main__":
    
    parser = ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='../plate-spring/spring/', help='dataset')
    parser.add_argument('--n_epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=lambda x: torch.device(x), default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--crop', type=int, default=3200)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--load', type=str, default=None, help='checkpoint to load')
    
    # Specific inference arguments
    parser.add_argument('--input_path', type=str, required=True, help='path to input audio file')
    parser.add_argument('--max_length', type=int, required=True, help='maximum length of the input audio')
    parser.add_argument('--stereo', type=bool, default=False, help='flag to indicate if the audio is stereo or mono')
    parser.add_argument('--tail', type=bool, default=False, help='flag to indicate if tail padding is required')
    parser.add_argument('--width', type=float, required=True, help='width parameter for the model')
    parser.add_argument('--c0', type=float, required=True, help='c0 parameter for the model')
    parser.add_argument('--c1', type=float, required=True, help='c1 parameter for the model')
    parser.add_argument('--gain_dB', type=float, required=True, help='gain in dB for the model')
    parser.add_argument('--mix', type=float, required=True, help='mix parameter for the model')
    args = parser.parse_args()

    make_inference(args)
