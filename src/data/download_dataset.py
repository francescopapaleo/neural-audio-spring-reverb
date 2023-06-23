import os
import subprocess

try:
    # Create a new directory
    os.makedirs("../datasets", exist_ok=True)

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
