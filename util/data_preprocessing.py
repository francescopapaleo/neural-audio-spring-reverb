import os
import h5py
import numpy as np

from config import DATASET


with h5py.File(os.path.join(DATASET,'dry_train_subset.h5'), 'r') as f:
    # Print the list of datasets in the file
    print(list(f.keys()))

with h5py.File(os.path.join(DATASET,'wet_train_subset.h5'), 'r') as f:
    # Print the list of datasets in the file
    print(list(f.keys()))


with h5py.File(os.path.join(DATASET,'dry_val_subset.h5'), 'r') as f:
    # Print the list of datasets in the file
    print(list(f.keys()))

with h5py.File(os.path.join(DATASET,'wet_val_subset.h5'), 'r') as f:
    # Print the list of datasets in the file
    print(list(f.keys()))
 
with h5py.File(os.path.join(DATASET,'dry_test_subset.h5'), 'r') as f:
    # Print the list of datasets in the file
    print(list(f.keys()))

with h5py.File(os.path.join(DATASET,'wet_test_subset.h5'), 'r') as f:
    # Print the list of datasets in the file
    print(list(f.keys()))