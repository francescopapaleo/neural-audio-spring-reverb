import h5py
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import os

seed = 42

# Main class that retrieves the entire dataset
class DataRetriever(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def retrieve_data(self):
        x_train_path = f"{self.data_folder}/dry_train.h5"
        y_train_path = f"{self.data_folder}/wet_train.h5"
        x_val_test_path = f"{self.data_folder}/dry_val_test.h5"
        y_val_test_path = f"{self.data_folder}/wet_val_test.h5"

        with h5py.File(x_train_path, 'r') as f:
            x_train = f['Xtrain'][:]

        with h5py.File(y_train_path, 'r') as f:
            y_train = f['Ytrain_0'][:]

        with h5py.File(x_val_test_path, 'r') as f:
            x_val_test = f['Xvalidation'][:]

        with h5py.File(y_val_test_path, 'r') as f:
            y_val_test = f['Yvalidation_0'][:]

        return x_train, y_train, x_val_test, y_val_test


class SubsetGenerator(DataRetriever):
    def __init__(self, data_folder, subset_size):
        super().__init__(data_folder)
        self.subset_size = subset_size

    @staticmethod
    def generate_idx(data, subset_size):
        # Generate a list of indices based on the seed
        np.random.seed(seed)
        idx = np.random.choice(np.arange(len(data)), size=subset_size, replace=False)
        return idx

    def select_random_samples(self, data, selected_idx):
        # Select the samples corresponding to the provided indices
        selected_samples = data[selected_idx]
        return selected_samples

    def save_subset(self, x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset, save_folder):
        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        with h5py.File(os.path.join(save_folder, 'x_train_subset.h5'), 'w') as f:
            f.create_dataset('Xtrain_subset', data=x_train_subset)

        with h5py.File(os.path.join(save_folder, 'y_train_subset.h5'), 'w') as f:
            f.create_dataset('Ytrain_0_subset', data=y_train_subset)

        with h5py.File(os.path.join(save_folder, 'x_val_test_subset.h5'), 'w') as f:
            f.create_dataset('Xvalidation_subset', data=x_val_test_subset)

        with h5py.File(os.path.join(save_folder, 'y_val_test_subset.h5'), 'w') as f:
            f.create_dataset('Yvalidation_0_subset', data=y_val_test_subset)
            
    def retrieve_subset(self):
        x_train, y_train, x_val_test, y_val_test = self.retrieve_data()

        # Generate indices for each data subset
        train_idx = self.generate_idx(x_train, self.subset_size)
        val_test_idx = self.generate_idx(x_val_test, self.subset_size // 2)  # Assuming you want half the size for val/test
        
        # Select samples
        x_train_subset = self.select_random_samples(x_train, train_idx)
        y_train_subset = self.select_random_samples(y_train, train_idx)
        x_val_test_subset = self.select_random_samples(x_val_test, val_test_idx)
        y_val_test_subset = self.select_random_samples(y_val_test, val_test_idx)

        # Save the subsets to the desired folder
        self.save_subset(x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset, 'dataset_subset')

        return x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset

class SubsetRetriever(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def concatenate_samples(self, data):
        # Concatenate samples along the first dimension
        concatenated_data = np.concatenate(data, axis=0)
        reshape_data = concatenated_data.reshape(1, -1)
        return reshape_data

    def retrieve_data(self, concatenate=True):
        x_train_path = f"{self.data_folder}/x_train_subset.h5"
        y_train_path = f"{self.data_folder}/y_train_subset.h5"
        x_val_test_path = f"{self.data_folder}/x_val_test_subset.h5"
        y_val_test_path = f"{self.data_folder}/y_val_test_subset.h5"

        with h5py.File(x_train_path, 'r') as f:
            x_train_subset = f['Xtrain_subset'][:]

        with h5py.File(y_train_path, 'r') as f:
            y_train_subset = f['Ytrain_0_subset'][:]

        with h5py.File(x_val_test_path, 'r') as f:
            x_val_test_subset = f['Xvalidation_subset'][:]

        with h5py.File(y_val_test_path, 'r') as f:
            y_val_test_subset = f['Yvalidation_0_subset'][:]

        if concatenate:
            # Concatenate samples
            x_train_concatenated = self.concatenate_samples(x_train_subset)
            y_train_concatenated = self.concatenate_samples(y_train_subset)
            x_val_test_concatenated = self.concatenate_samples(x_val_test_subset)
            y_val_test_concatenated = self.concatenate_samples(y_val_test_subset)

            return x_train_concatenated, y_train_concatenated, x_val_test_concatenated, y_val_test_concatenated
        else:
            return x_train_subset, y_train_subset, x_val_test_subset, y_val_test_subset

class AudioDataRetriever(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def retrieve_data(self):
        audio_files = os.listdir(self.data_folder)
        audio_data = []

        for file in audio_files:
            filepath = os.path.join(self.data_folder, file)
            waveform, sample_rate = torchaudio.load(filepath)

            # Normalize to mono (single channel)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            audio_data.append(waveform.numpy())

        return np.array(audio_data)
    
    