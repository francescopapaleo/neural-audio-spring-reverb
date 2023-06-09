""" RT 60 measurement using Schroeder's method.

Reference:
----------
Modified version of the source code from pyroomacoustics:
https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/experimental/rt60.py


.. [1] M. R. Schroeder, "New Method of Measuring Reverberation Time,"
    J. Acoust. Soc. Am., vol. 37, no. 3, pp. 409-412, Mar. 1968.
"""

import numpy as np
from scipy.io import wavfile
from pathlib import Path
from argparse import ArgumentParser
from utils.plotter import plot_rt60

eps = 1e-15


def measure_rt60(h, sample_rate, decay_db=60, rt60_tgt=None, plot=True, file_name=None):
    """
    Arguments
    ----------
    h (array_like): The impulse response.
    sample_rate (float or int): The sampling frequency of h.
    decay_db (float or int, optional): The decay in decibels for which we estimate the time. 
        Defaults to 60 dB. This value is used to estimate the time taken for the sound energy to decay by this amount. 
        Often in practice we measure the RT20 or RT30 and extrapolate to RT60.
    rt60_tgt (float, optional): This parameter can be used to indicate a target RT60 to which we want to compare the estimated value.
        Defaults to None.
    plot (bool, optional): If set to ``True``, the power decay and different estimated values will
        be plotted and saved as a png file. Defaults to True.
    
    Returns
    -------
    float: The estimated RT60 value in seconds.
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
        i_decay = np.min(np.where(-decay_db - energy_db > 0)[0])
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
            return np.arange(x.shape[0]) / fs - i_5db / fs

        T = get_time(energy_db, fs)
        plot_rt60(T, energy_db, e_5db, est_rt60, rt60_tgt, file_name)

    return est_rt60


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--input', type=str, required=True, help='path to input audio file')
    
    args = parser.parse_args()

    fs, data = wavfile.read(args.input)
    x = data.astype('float32')

    est_rt60 = measure_rt60(x, fs, decay_db=60, rt60_tgt=None , plot=True, file_name=Path(args.input).stem)
    print(f"The RT60 is {est_rt60 * 1000:.0f} ms")