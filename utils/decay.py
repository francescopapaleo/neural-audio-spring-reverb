import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from argparse import ArgumentParser
from pathlib import Path

def decay_db(h, fs, input_file):
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax = [ax]

    t = np.linspace(0, len(energy_db) / fs, num=len(energy_db))
    ax[0].plot(t, energy_db.T, linestyle='-', alpha=0.8)
    ax[0].set_title("Energy Decay Curve (dB)")
    ax[0].set_ylabel('Energy [dB]', rotation=90, labelpad=20)
    ax[0].grid(c="lightgray")
    ax[0].set_axisbelow(True)
    ax[0].set_xlabel("Time [s]")

    # Extract the filename without extension and append 'decay.png'
    input_filename = Path(input_file).stem
    output_filename = f"{input_filename}_decay.png"

    plt.savefig(Path('./plots') / output_filename)

    # Find the index where energy_db first drops below -60 dB
    indices_below_60dB = np.where(energy_db <= -60)[0]
    if len(indices_below_60dB) > 0:
        time_at_60dB = t[indices_below_60dB[0]]
        print(f"The curve first reaches -60 dB at t = {time_at_60dB * 1000:.0f} ms")
    else:
        print("The curve never reaches -60 dB")

    return energy_db

def main(args):

    fs, data = wavfile.read(args.input)

    decay_db(x, fs, args.input)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--input', type=str, required=True, help='path to input audio file')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate')
    
    args = parser.parse_args()

    main(args)
