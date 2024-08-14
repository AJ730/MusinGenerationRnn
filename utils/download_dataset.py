# Helper functions
import os
import pathlib
import urllib
import zipfile


def download_and_extract_maestro(data_dir: str, url: str):
    # Convert data_dir to a pathlib.Path object if it's a string
    print("Function called")  # Debugging print statement

    data_dir = pathlib.Path(data_dir)

    if not data_dir.exists():
        # Download the dataset
        zip_path = pathlib.Path('data/maestro-v2.0.0-midi.zip')
        if not zip_path.exists():
            print("Downloading the dataset...")
            urllib.request.urlretrieve(url, zip_path)

        # Extract the ZIP file
        print("Extracting the dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir.parent)

        # Remove the ZIP file after extraction
        os.remove(zip_path)
        print("Dataset downloaded and extracted successfully.")
    else:
        print("Dataset already exists.")

