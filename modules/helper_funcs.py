import torch
import matplotlib.pylot as plt
import numpy as np
from torchimport nn
import os
import zipfile
from pathlib import Path
import requests

def download_data(source: str,
                  destination: str,
                  remove_source: bool=True) -> Path:
    """downloads and unzips dataset to given destination.

    Args:
    source: link to zipped file.
    destination: directory in which to unzip data.
    remove_source: whether to remove the source after data is extracted.

    Returns:
    pathlib.Path for data
    """

    # path to data folder
    data_path = Path('data/')
    image_path = data_path / destination
    
    # check if image folder exists, if not download it
    if image_path.is_dir():
        print(f'directory already exists.')
    else:
        print(f"{image_path} does not exist, creating it")
        image_path.mkdir(parents=True, exist_ok=True)
    
        # download 
        target_file = Path(source).name
        with open(data_path / target_file, 'wb') as f:
            request = requests.get(source)
            print(f'downloading {target_file} from {source}')
            f.write(request.content)
    
        # unzip
        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
            print(f'unzipping {target_file}')
            zip_ref.extractall(image_path)
    
        # remove zip file
        if remove_source:
          os.remove(data_path / target_file)
    
    return image_path
