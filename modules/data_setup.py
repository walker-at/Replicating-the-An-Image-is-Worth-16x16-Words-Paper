import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
  train_dir: str,
  test_dir: str,
  transform: transforms.Compose,
  batch_size: int,
  num_workers: int=NUM_WORKERS
):
  """Creates PyTorch Dataloaders from training and testing directory paths

  Args:
    train_dir: path to training dir
    test_dir: path to testing dir
    transform: torchvision transforms to perform on data
    batch_size: samples per batch in the Dataloaders
    num_workers: num of workers per Dataloader

  Returns:
    a tuple consisting of (train_dataloader, test_dataloader, class_names).
  """
  # Use ImageFolder to create datasets
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
  )
  test_dataloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
  )

  return train_dataloader, test_dataloader, class_names
