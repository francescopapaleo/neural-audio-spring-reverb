import h5py
import numpy as np
import os

class DataLoader:
    def __init__(self, data_folder):
        self.data_folder = data_folder

    def load_data(self, dataset_name):
        file_list = [f for f in os.listdir(self.data_folder) if f.endswith('.h5') and dataset_name in f]
        data_list = []

        for file_name in file_list:
            file_path = os.path.join(self.data_folder, file_name)
            with h5py.File(file_path, 'r') as f:
                dataset_key = list(f.keys())[0]  # assumes only one dataset per file
                data = f[dataset_key][:]
                data_list.append(data)

        return data_list

    def get_datasets(self):
        x_train = self.load_data('Xtrain_subset')
        y_train = self.load_data('Ytrain_0_subset')
        x_valid = self.load_data('Xvalidation_subset')
        y_valid = self.load_data('Yvalidation_0_subset')
        x_test = self.load_data('Xtest_subset')
        y_test = self.load_data('Ytest_0_subset')

        return x_train, y_train, x_valid, y_valid, x_test, y_test
