import subprocess
from pathlib import Path

"""This script downloads the selected dataset.
"""


def download_file(destination, url):
    try:
        destination = Path(destination)
        subprocess.run(["wget", "-P", str(destination), url], check=True)
    except FileNotFoundError:
        print(
            "Error: The 'wget' command could not be found."
            "Please ensure 'wget' is installed and available."
        )
        raise
    except subprocess.CalledProcessError:
        print(
            "Error: The 'wget' command failed."
            "Please check the URL or your internet connection."
        )
        raise


def unzip_file(zip_path, output_dir):
    try:
        zip_path = Path(zip_path)
        output_dir = Path(output_dir)
        subprocess.run(["unzip", str(zip_path), "-d", str(output_dir)], check=True)
    except FileNotFoundError:
        print(
            "Error: The 'unzip' command could not be found."
            "Please ensure 'unzip' is installed and available."
        )
        raise
    except subprocess.CalledProcessError:
        print("Error: The 'unzip' command failed. Please check the zip file.")
        raise


def download_data(args):
    args.data_dir = Path(args.data_dir)

    if args.dataset == "springset":
        try:
            # Create a subdirectory for 'springset'
            springset_dir = args.data_dir / "springset"
            springset_dir.mkdir(parents=True, exist_ok=True)

            download_file(
                args.data_dir,
                "https://zenodo.org/record/3746119/files/plate-spring.zip",
            )
            unzip_file(args.data_dir / "plate-spring.zip", args.data_dir)
            (args.data_dir / "plate-spring.zip").unlink()  # Delete zip file

            for file in (args.data_dir / "plate/").glob("*"):
                file.unlink()  # Delete the files within 'plate' directory
            (args.data_dir / "plate").rmdir()  # Delete the 'plate' subdir
            (args.data_dir / "spring").rename(
                args.data_dir / "springset"
            )  # Rename 'spring' to 'springset'
        except Exception as e:
            print("An unexpected error occurred:", e)

    elif args.dataset == "egfxset":
        try:
            # Create a subdirectory for 'egfxset'
            egfxset_dir = args.data_dir / "egfxset"  # Convert to Path object
            egfxset_dir.mkdir(parents=True, exist_ok=True)

            download_file(
                egfxset_dir, "https://zenodo.org/record/7044411/files/Clean.zip"
            )
            unzip_file(egfxset_dir / "Clean.zip", egfxset_dir)
            (egfxset_dir / "Clean.zip").unlink()  # Delete the zip file

            download_file(
                egfxset_dir, "https://zenodo.org/record/7044411/files/Spring-Reverb.zip"
            )
            unzip_file(egfxset_dir / "Spring-Reverb.zip", egfxset_dir)
            (egfxset_dir / "Spring-Reverb.zip").unlink()  # Delete the zip file
        except Exception as e:
            print("An unexpected error occurred:", e)

    else:
        raise ValueError(
            "Dataset not found. Please select from 'springset' or 'egfxset'."
        )
