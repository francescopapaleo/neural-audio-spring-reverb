from pathlib import Path

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


eps = 1e-15
args = parser.parse_args()

"""
RT60 Measurement Routine
========================

Automatically determines the reverberation time of an impulse response
using the Schroeder method [1].

References
----------
Modified version of the source code from pyroomacoustics:
https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/experimental/rt60.py


.. [1] M. R. Schroeder, "New Method of Measuring Reverberation Time,"
    J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.
"""


def measure_rt60(h, sample_rate, decay_db=60, plot=False, rt60_tgt=None, folder='results'):
    """
    Analyze the RT60 of an impulse response. Optionaly plots some useful information.

    Parameters
    ----------
    h: array_like
        The impulse response.
    fs: float or int, optional
        The sampling frequency of h (default to 1, i.e., samples).
    decay_db: float or int, optional
        The decay in decibels for which we actually estimate the time. Although
        we would like to estimate the RT60, it might not be practical. Instead,
        we measure the RT20 or RT30 and extrapolate to RT60.
    plot: bool, optional
        If set to ``True``, the power decay and different estimated values will
        be plotted (default False).
    rt60_tgt: float
        This parameter can be used to indicate a target RT60 to which we want
        to compare the estimated value.
    """

    
    h = np.array(h)
    fs = float(sample_rate)

    # The power of the impulse response in dB
    power = h**2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    try:
        # remove the possibly all zero tail
        i_nz = np.max(np.where(energy > 0)[0])
        energy = energy[:i_nz]
        energy_db = 10 * np.log10(energy)
        energy_db -= energy_db[0]

        # -5 dB headroom
        i_5db = np.min(np.where(-5 - energy_db > 0)[0])
        e_5db = energy_db[i_5db]
        t_5db = i_5db / fs

        # after decay
        i_decay = np.min(np.where(-5 - decay_db - energy_db > 0)[0])
        t_decay = i_decay / fs

        # compute the decay time
        decay_time = t_decay - t_5db
        est_rt60 = (60 / decay_db) * decay_time
    except:
        est_rt60 = np.array(0.0)

    if plot:

        # Remove clip power below to minimum energy (for plotting purpose mostly)
        energy_min = energy[-1]
        energy_db_min = energy_db[-1]
        power[power < energy[-1]] = energy_min
        power_db = 10 * np.log10(power)
        power_db -= np.max(power_db)

        # time vector
        def get_time(x, fs):
            time_array = np.arange(x.shape[0]) / fs
            return time_array - time_array[i_5db]

        T = get_time(power_db, fs)

        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(get_time(energy_db, sample_rate), energy_db, label="Energy")

        # now the linear fit
        ax.plot([0, est_rt60], [e_5db, -65], "--", label="Linear Fit")
        ax.plot(T, np.ones_like(T) * -60, "--", label="-60 dB")
        ax.vlines(est_rt60, energy_db[-1], 0, linestyles="dashed", label="Estimated RT60")

        if rt60_tgt is not None:
            ax.vlines(rt60_tgt, energy_db[-1], 0, label="Target RT60")

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (dB)')
        ax.set_title('RT60 Measurement')
        ax.grid(True)

        ax.legend()

        fig.savefig(Path(folder) / 'rt60.png')
        plt.close(fig)

    return est_rt60


def test_rt60(ir, fs=16000):
    """
    Very basic test that runs the function and checks that the value
    returned with different sampling frequencies are correct.
    """

    t60_samples = measure_rt60(ir)
    t60_s = measure_rt60(ir, fs)

    assert abs(t60_s - t60_samples / fs) < eps

    t30_samples = measure_rt60(ir, decay_db=30)
    t30_s = measure_rt60(ir, fs, decay_db=30)

    assert abs(t30_s - t30_samples / fs) < eps

    print(f"RT60 (samples): {t60_samples}")
    print(f"RT60 (seconds): {t60_s}")
    print(f"RT30 (samples): {t30_samples}")
    print(f"RT30 (seconds): {t30_s}")


def test_rt60_plot(ir):
    """
    Simple run of the plot without any output.

    Check for runtime errors only.
    """

    import matplotlib

    matplotlib.use("Agg")

    measure_rt60(ir, plot=True)
    measure_rt60(ir, plot=True, rt60_tgt=0.3) 


def main():
     
    args = parser.parse_args()

    data, file_fs = wavfile.read(args.input_file, always_2d=True)
    channel = 0  # Select the left channel, change to 1 for the right channel
    data = data[:, channel]

    fs = args.fs if args.fs is not None else file_fs
    rt60 = measure_rt60(data, fs=fs, decay_db=args.decay_db, plot=args.plot, rt60_tgt=args.rt60_tgt)

    if args.plot:
        plt.show()

    print(f"The RT60 is {rt60 * 1000:.0f} ms")

if __name__ == "__main__":
    main()
