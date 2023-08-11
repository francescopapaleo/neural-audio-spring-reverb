# data/download_dataset.py

import mirdata
import os
from configurations import parse_args

if __name__ == "__main__":

    args = parse_args()

    try:
        # Create a new directory
        os.makedirs(args.datadir, exist_ok=True)

        # Download the file
        try:
            egfx = mirdata.initialize('egfxset', data_home='../datasets/egfxset/') # initializes the dataset
            egfx.download(partial_download=['metadata', 'clean', 'spring-Reverb']) # downloads the dataset
        except:
            print("Error downloading the dataset")

    except OSError:
        print("Creation of the directory %s failed" % args.datadir)