import numpy as np
import argparse
import soundfile as sf
from pyroomacoustics.experimental.rt60 import measure_rt60

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="RT60 Measurement Routine")

    parser.add_argument("file", help="Path to the WAV file")
    parser.add_argument(
        "--fs", type=float, default=1, help="Sampling frequency of the impulse response"
    )
    parser.add_argument(
        "--decay_db",
        type=float,
        default=60,
        help="Decay in decibels for which to estimate the time",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the power decay and different estimated values",
    )
    parser.add_argument(
        "--rt60_tgt",
        type=float,
        help="Target RT60 to which to compare the estimated value",
    )

    args = parser.parse_args()

    try:
        # Read the WAV file
        data, sample_rate = sf.read(args.file)

        # Extract mono audio data if it contains multiple channels
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Measure the RT60
        rt60 = measure_rt60(
            data,
            fs=args.fs,
            decay_db=args.decay_db,
            plot=args.plot,
            rt60_tgt=args.rt60_tgt,
        )
        print("Estimated RT60:", rt60)

        # Show the plot
        if args.plot:
            plt.show()

    except FileNotFoundError:
        print(f"File not found: {args.file}")
    except Exception as e:
        print("An error occurred:", str(e))


if __name__ == "__main__":
    main()
