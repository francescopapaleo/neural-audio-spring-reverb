""" Run this script to download the datasets.

"""

import os
import subprocess
from configurations import parse_args

if __name__ == "__main__":

    args = parse_args()

    if springset is not None:
        try:
            # Create a new directory
            os.makedirs(args.datadir, exist_ok=True)

            # Download the file
            try:
                subprocess.run(["wget", "-P", "../datasets/", "https://zenodo.org/record/3746119/files/plate-spring.zip"], check=True)
            except FileNotFoundError:
                print("Error: The 'wget' command could not be found. Please ensure 'wget' is installed and available in your system's path.")
                raise
            except subprocess.CalledProcessError:
                print("Error: The 'wget' command failed. Please check the URL or your internet connection.")
                raise

            # Unzip the file
            try:
                subprocess.run(["unzip", "../datasets/plate-spring.zip", "-d", "../datasets/plate-spring"], check=True)
            except FileNotFoundError:
                print("Error: The 'unzip' command could not be found. Please ensure 'unzip' is installed and available in your system's path.")
                raise
            except subprocess.CalledProcessError:
                print("Error: The 'unzip' command failed. Please check the zip file.")
                raise
        except Exception as e:
            print("An unexpected error occurred:", e)

    if egfxset is not None:
        try:
            # Create a new directory
            os.makedirs(args.datadir, exist_ok=True)

            # Download the file
            try:
                subprocess.run(["wget", "-P", "../datasets/", "https://zenodo.org/record/7044411/files/Clean.zip"], check=True)
                subprocess.run(["wget", "-P", "../datasets/", "https://zenodo.org/record/7044411/files/Spring-Reverb.zip"], check=True)
            except FileNotFoundError:
                print("Error: The 'wget' command could not be found. Please ensure 'wget' is installed and available in your system's path.")
                raise
            except subprocess.CalledProcessError:
                print("Error: The 'wget' command failed. Please check the URL or your internet connection.")
                raise

            # Unzip the file
            try:
                subprocess.run(["unzip", "../datasets/egfxset.zip", "-d", "../datasets/egfxset"], check=True)
            except FileNotFoundError:
                print("Error: The 'unzip' command could not be found. Please ensure 'unzip' is installed and available in your system's path.")
                raise
            except subprocess.CalledProcessError:
                print("Error: The 'unzip' command failed. Please check the zip file.")
                raise
        except Exception as e:
            print("An unexpected error occurred:", e)