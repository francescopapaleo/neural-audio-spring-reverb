
"""
Test code for RT60 measurement routine
Modified version of the source code from pyroomacoustics:
https://github.com/LCAV/pyroomacoustics/blob/master/pypyroomacoustics/experimental/tests/test_measure_rt60.py
"""

import numpy as np
import pyroomacoustics as pra
from utils.rt60_measure import measure_rt60  # Import the modified version of measure_rt60

eps = 1e-15

e_abs = 1.0 - (1.0 - 0.35) ** 2
room = pra.ShoeBox([10, 7, 6], fs=16000, materials=pra.Material(e_abs), max_order=17)
room.add_source([3, 2.5, 1.7])
room.add_microphone_array(pra.MicrophoneArray(np.array([[7, 3.7, 1.1]]).T, room.fs))
room.compute_rir()

ir = room.rir[0][0]


def test_rt60():
    """
    Very basic test that runs the function and checks that the value
    returned with different sampling frequencies are correct.
    """

    t60_samples = measure_rt60(ir)  # Use the modified version of measure_rt60
    t60_s = measure_rt60(ir, fs=room.fs)  # Use the modified version of measure_rt60

    assert abs(t60_s - t60_samples / room.fs) < eps

    t30_samples = measure_rt60(ir, decay_db=30)  # Use the modified version of measure_rt60
    t30_s = measure_rt60(ir, decay_db=30, fs=room.fs)  # Use the modified version of measure_rt60

    assert abs(t30_s - t30_samples / room.fs) < eps

    print(f"RT60 (samples): {t60_samples}")
    print(f"RT60 (seconds): {t60_s}")
    print(f"RT30 (samples): {t30_samples}")
    print(f"RT30 (seconds): {t30_s}")


def test_rt60_plot():
    """
    Simple run of the plot without any output.

    Check for runtime errors only.
    """

    import matplotlib

    matplotlib.use("Agg")

    measure_rt60(ir, plot=True)  # Use the modified version of measure_rt60
    measure_rt60(ir, plot=True, rt60_tgt=0.3)  # Use the modified version of measure_rt60


if __name__ == "__main__":
    test_rt60()
    test_rt60_plot()