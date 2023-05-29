from pathlib import Path
import torch
import torchaudio
import torchsummary
import argparse
import auraloss
import scipy.signal

from utils.rt60 import measure_rt60
from tcn import TCN, causal_crop

torch.backends.cudnn.benchmark = True

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
    model = model.to(args.device)  # move the model to the right device

    torchsummary.summary(model, [(1,65536), (1,2)], device=args.device)

    print("## Initializing metrics...")

    # Initialize lists for storing metric values
    mse = torch.nn.MSELoss()
    l1 = torch.nn.L1Loss()
    stft = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048],
        win_lengths=[32, 128, 512, 2048],
        hop_sizes=[16, 64, 256, 1024])
    esr = auraloss.time.ESRLoss()
    dc = auraloss.time.DCLoss()
    snr = auraloss.time.SNRLoss()

    test_results = {
        "mse": [],
        "l1": [],
        "stft": [],
        "esr": [],
        "dc": [],
        "snr": []
    }

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    print("## Testing...")
    model.eval()
    with torch.no_grad():
        for input, target in testloader:

            input = input.float().to(args.device)
            target = target.float().to(args.device)
            c = torch.tensor([0.0, 0.0], device=device).view(1,1,-1)

            rf = model.compute_receptive_field()
            input_pad = torch.nn.functional.pad(input, (rf-1, 0))

            output = model(input_pad)

            # Calculate the metrics
            test_results['mse'].append(mse(output, target))
            test_results['l1'].append(l1(output, target))
            test_results['stft'].append(stft(output, target))
            test_results['esr'].append(esr(output, target))
            test_results['dc'].append(dc(output, target))
            test_results['snr'].append(snr(output, target))

    # Save metric data
    with open(Path(args.results_dir) / 'eval_metrics.pkl', 'wb') as f:
        pickle.dump(test_results, f)
    
    # Normalize
    for name, values in test_results.items():
        values = (values - np.mean(values)) / np.std(values)

    print('Saving audio files...')
    ofile = Path(args.results_dir) / 'eval_output.wav'
    o_float = output.view(1,-1).cpu().float()
    torchaudio.save(ofile, o_float, sample_rate)
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
    
    make_inference()
