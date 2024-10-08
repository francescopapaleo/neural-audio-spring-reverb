from pathlib import Path
import h5py
import torch
import numpy as np


class SpringDataset(torch.utils.data.Dataset):
    """
    Attributes:
    -----------
        root_dir (pathlib.Path): The root directory where the data files are located.
        split (str, optional): The data split to use (i.e. 'train', 'test', or 'validation').
            If not provided, all data in the root directory will be used.
        transform (callable, optional): Optional transform to apply to the data samples.

    Methods:
    --------
        __len__(): Returns the length of the dataset.
        __getitem__(index: int): Returns the data at a given index.
        load_data(): Loads data from the files specified in the root directory.
        print_info(): Prints information about the loaded data files and their contents.
        print_index_info(index: int): Prints information about a specific index in the data.

    Reference:
    ----------
    @dataset{martinez_ramirez_marco_a_2019_3746119,
    author       = {Martinez Ramirez, Marco A and
                    Benetos, Emmanouil and
                    Reiss, Joshua D},
    title        = {{Modeling plate and spring reverberation using a
                    DSP-informed deep neural network}},
    month        = oct,
    year         = 2019,
    publisher    = {Zenodo},
    doi          = {10.5281/zenodo.3746119},
    url          = {https://doi.org/10.5281/zenodo.3746119}
    }
    """

    def __init__(self, root_dir, split=None, transforms=None):
        super(SpringDataset, self).__init__()
        self.root_dir = Path(root_dir) / "springset"
        self.split = split
        self.seed = torch.seed()

        self.file_list = list(self.root_dir.glob("**/*.h5"))
        self.dry_file = [
            f for f in self.file_list if "dry" in f.stem and self.split in f.stem
        ][0]
        self.wet_file = [
            f for f in self.file_list if "wet" in f.stem and self.split in f.stem
        ][0]
        self.dry_data = None
        self.wet_data = None

        # Check if the files are not empty
        if self.dry_file.stat().st_size == 0 or self.wet_file.stat().st_size == 0:
            raise ValueError("Data files are empty. Please check the data integrity.")

        print(f"Found {len(self.file_list)} files in {self.root_dir}")
        print(
            f"Using {self.dry_file.name} and {self.wet_file.name} for {self.split} split."
        )

        self.load_data()
        self.index = {
            i: (self.dry_file.name, self.wet_file.name, i)
            for i in range(len(self.dry_data))
        }

        self.transforms = transforms

    def __len__(self):
        # The length should be the number of items in the data
        return len(self.index)

    def __getitem__(self, index):
        # Returns a tuple of numpy arrays
        x, y = self.dry_data[index], self.wet_data[index]
        x, y = x.reshape(1, -1), y.reshape(1, -1)

        # Convert numpy arrays to tensors
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        # Transforms are provided as a list
        if self.transforms:
            for transform in self.transforms:
                x = transform(x)
                y = transform(y)

        return x, y

    def load_data(self):
        with (
            h5py.File(self.dry_file, "r") as f_dry,
            h5py.File(self.wet_file, "r") as f_wet,
        ):
            dry_key = list(f_dry.keys())[0]
            wet_key = list(f_wet.keys())[0]
            self.dry_data = f_dry[dry_key][:].astype(np.float32)
            self.wet_data = f_wet[wet_key][:].astype(np.float32)

    def normalize_data(self):
        # Concatenate dry and wet data along the first dimension (assuming the data shape is [n_samples, n_features])
        all_data = np.concatenate([self.dry_data, self.wet_data], axis=0)

        # Compute the maximum absolute value across all data
        max_val = np.max(np.abs(all_data))

        # Normalize the dry and wet data using the maximum value
        self.dry_data = self.dry_data / max_val
        self.wet_data = self.wet_data / max_val

    def print_info(self):
        print(f"Dry File: {self.dry_file}")
        print(f"Wet File: {self.wet_file}")
        print(f"Dry data shape: {self.dry_data.shape}")
        print(f"Wet data shape: {self.wet_data.shape}")

        dry_samples_total = np.sum([sample.shape[0] for sample in self.dry_data])
        wet_samples_total = np.sum([sample.shape[0] for sample in self.wet_data])
        dry_minutes_total = dry_samples_total / 16000 / 60
        wet_minutes_total = wet_samples_total / 16000 / 60

        print(f"Dry samples total: {dry_samples_total}")
        print(f"Wet samples total: {wet_samples_total}")
        print(f"Dry minutes total: {dry_minutes_total:.2f}")
        print(f"Wet minutes total: {wet_minutes_total:.2f}")

        print("(item_n, samples_per_item, channels)")

    def print_index_info(self, index):
        """Prints the information about a specific index tuple."""
        print(f"Index: {index}")
        print(f"Index data: {self.index[index]}")


def correct_dc_offset(tensor):
    return tensor - torch.mean(tensor)


def peak_normalize(tensor):
    tensor /= torch.max(torch.abs(tensor))

    return tensor


TRANSFORMS = [correct_dc_offset, peak_normalize]


def load_springset(datadir, batch_size, train_ratio=0.6, num_workers=4):
    """Load and split the dataset"""
    trainset = SpringDataset(root_dir=datadir, split="train", transforms=TRANSFORMS)
    train_size = int(train_ratio * len(trainset))
    valid_size = len(trainset) - train_size
    train, valid = torch.utils.data.random_split(trainset, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(
        train, batch_size, num_workers=num_workers, shuffle=True, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )

    testset = SpringDataset(root_dir=datadir, split="test", transforms=TRANSFORMS)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size, num_workers=num_workers, drop_last=True
    )

    return train_loader, valid_loader, test_loader
