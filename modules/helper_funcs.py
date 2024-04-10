import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import os
import zipfile
from pathlib import Path
import requests

def get_data(source: str,
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

# set seeds
def set_seeds(seed: int=42):
    """Sets random sets for PyTorch.

    Args:
        seed (int, optional): Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """
    Saves a PyTorch model to a target directory.
  
    Args:
      model: PyTorch model to save.
      target_dir: Where to save to.
      model_name: Filename for the model.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)
  
    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt")
    model_save_path = target_dir_path / model_name
  
    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
