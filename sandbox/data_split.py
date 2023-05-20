import h5py
import numpy as np
import soundfile as sf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def h5_to_audio(h5_filename, wav_filename):
    with h5py.File(os.path.join(DATASET, h5_filename), 'r') as f:
        key = list(f.keys())[0]  
        data = np.array(f[key])
        sf.write(os.path.join(DATASET, wav_filename), data, sample_rate)

# Convert the files
h5_to_audio('x_train_subset.h5', 'x_train_subset.wav')
h5_to_audio('y_train_subset.h5', 'y_train_subset.wav')
h5_to_audio('x_val_test_subset.h5', 'x_val_test_subset.wav')
h5_to_audio('y_val_test_subset.h5', 'y_val_test_subset.wav')
