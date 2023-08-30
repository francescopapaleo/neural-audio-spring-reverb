""" 
Run this script to download the datasets.
"""

from pathlib import Path
import subprocess
from configurations import parse_args

def download_file(destination, url):
    try:
        subprocess.run(["wget", "-P", str(destination), url], check=True)
    except FileNotFoundError:
        print("Error: The 'wget' command could not be found. Please ensure 'wget' is installed and available in your system's path.")
        raise
    except subprocess.CalledProcessError:
        print("Error: The 'wget' command failed. Please check the URL or your internet connection.")
        raise

def unzip_file(zip_path, output_dir):
    try:
        subprocess.run(["unzip", str(zip_path), "-d", str(output_dir)], check=True)
    except FileNotFoundError:
        print("Error: The 'unzip' command could not be found. Please ensure 'unzip' is installed and available in your system's path.")
        raise
    except subprocess.CalledProcessError:
        print("Error: The 'unzip' command failed. Please check the zip file.")
        raise

if __name__ == "__main__":

    args = parse_args()
    datadir = Path(args.datadir)

    if args.dataset == 'springset':
        try:
            # Create a new directory
            datadir.mkdir(parents=True, exist_ok=True)

            download_file(datadir, "https://zenodo.org/record/3746119/files/plate-spring.zip")
            unzip_file(datadir / "plate-spring.zip", datadir)

        except Exception as e:
            print("An unexpected error occurred:", e)

    if args.dataset == 'egfxset':
        try:
            # Create a subdirectory for 'egfxset'
            egfxset_dir = datadir / "egfxset"
            egfxset_dir.mkdir(parents=True, exist_ok=True)

            download_file(egfxset_dir, "https://zenodo.org/record/7044411/files/Clean.zip")
            unzip_file(egfxset_dir / "Clean.zip", egfxset_dir)

            download_file(egfxset_dir, "https://zenodo.org/record/7044411/files/Spring-Reverb.zip")
            unzip_file(egfxset_dir / "Spring-Reverb.zip", egfxset_dir)

        except Exception as e:
            print("An unexpected error occurred:", e)
