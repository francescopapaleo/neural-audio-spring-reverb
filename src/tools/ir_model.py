from logging import config
import numpy as np
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal

from src.tools.ir_signals import generate_reference
from src.networks.checkpoints import load_model_checkpoint
from src.inference import make_inference

# from src.tools.plotter import plot_ir_spectrogram, plot_waterfall


def plot_waterfall(waveform, title, sample_rate, args, stride=1):
    frequencies, times, Sxx = signal.spectrogram(
        waveform,
        fs=sample_rate,
        window="blackmanharris",
        nperseg=32,
        noverlap=16,
        scaling="spectrum",
        mode="magnitude",
    )
    # Convert magnitude to dB
    Sxx_dB = 20 * np.log10(Sxx)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    X, Y = np.meshgrid(frequencies, times[::stride])
    Z = Sxx_dB.T[::stride]

    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="inferno",
        edgecolor="none",
        alpha=0.8,
        linewidth=0,
        antialiased=False,
    )

    # Autoscale and add colorbar
    ax.autoscale()
    cbar = fig.colorbar(surf, ax=ax, pad=0.01, aspect=35, shrink=0.5)
    cbar.set_label("Magnitude (dB)")

    # Set labels and title
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time (seconds)")
    ax.set_zlabel("Magnitude (dB)")

    ax.set_xlim([frequencies[-1], frequencies[0]])
    ax.view_init(
        elev=10, azim=45, roll=None, vertical_axis="z"
    )  # Adjusts the viewing angle for better visualization
    plt.tight_layout()
    plt.savefig(title)


def plot_ir_spectrogram(signal, sample_rate, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    duration_seconds = len(signal) / sample_rate

    cax = ax.specgram(
        signal,
        NFFT=512,
        Fs=sample_rate,
        noverlap=256,
        cmap="hot",
        scale="dB",
        mode="magnitude",
        vmin=-100,
        vmax=0,
    )
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel(f"Time [sec] ({duration_seconds:.2f} s)")  # Label in seconds
    ax.grid(True)
    ax.set_title(title)

    cbar = fig.colorbar(mappable=cax[3], ax=ax, format="%+2.0f dB")
    cbar.set_label("Intensity [dB]")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved spectrogram plot to {save_path}")


def measure_model_ir(args):
    """
    Impulse response measurement of a trained model
    =========================================================
    1. Generate the analysis signals
    2. Make inference with the model on the sweep tone
    3. Convolve the sweep tone with the inverse filter
    4. Normalize the impulse response
    5. Plot the spectrogram
    6. Save the impulse response as a .wav file
    """
    print("Measure the impulse response of a trained model")

    model, _, _, config, _, _ = load_model_checkpoint(args)

    # Generate the reference signals
    sweep, inverse_filter, _ = generate_reference(args.duration, config["sample_rate"])
    # sweep:(240000,) || inverse_filter:(240000,)
    sweep = sweep.reshape(1, -1)

    # Make inference with the model on the sweep tone
    # args.input = torch.from_numpy(sweep).float()
    # print(f'input waveform: {input.shape}')
    args.input = sweep

    sweep_output = make_inference(args)

    # print(f"sweep_output:{sweep_output.shape} || inverse_filter:{inverse_filter.shape}")
    sweep_output = sweep_output.reshape(-1).numpy()

    # Normalize the sweep tone and the inverse filter
    sweep_output = sweep_output - np.mean(sweep_output)
    sweep_output /= np.max(np.abs(sweep_output))
    inverse_filter /= np.max(np.abs(inverse_filter))

    # print("post squeeze")
    # print(f"sweep_output:{sweep_output.shape} || inverse_filter:{inverse_filter.shape}")

    # Convolve the sweep tone with the inverse filter
    impulse_response = signal.convolve(
        sweep_output, inverse_filter, mode="full", method="direct"
    )
    print(f"IR: {impulse_response.shape}")

    # Normalize the impulse response
    impulse_response = impulse_response - np.mean(impulse_response)
    impulse_response /= np.max(np.abs(impulse_response))
    print(f"ir:{impulse_response.max(), impulse_response.min()}")

    # Plot spectrogram
    spectra_file = Path(args.plots_dir) / f"IR_{Path(args.checkpoint).stem}.png"
    plot_ir_spectrogram(
        impulse_response,
        config["sample_rate"],
        f"Model: {config['name']} Spectrogram",
        spectra_file,
    )

    # Plot the waterfall spectrogram
    waterfall_file = (
        Path(args.plots_dir) / f"IR_{Path(args.checkpoint).stem}_waterfall.png"
    )
    plot_waterfall(impulse_response, waterfall_file, config["sample_rate"], args, 1)

    ir_tensor = torch.from_numpy(impulse_response).unsqueeze(0).float()
    print(f"tensor:{ir_tensor.max(),ir_tensor.min()}")

    # Create the directory if it does not exist
    save_directory = Path(args.audio_dir) / "IR_models"
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    save_as = f"{save_directory}/{Path(args.checkpoint).stem}_IR.wav"
    torchaudio.save(
        save_as, ir_tensor, config["sample_rate"], bits_per_sample=config["bit_rate"]
    )
    print(
        f"Saved measured impulse response to {save_as}, sample rate: {config['sample_rate']}, bit depth: {config['bit_rate']}"
    )
