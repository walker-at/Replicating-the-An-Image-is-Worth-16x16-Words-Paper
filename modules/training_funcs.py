import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.Dataloader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model over one epoch.

    Puts PyTorch model into training mode then runs through 
    the forward pass, loss calculation, and optimizer step).

    Args: 
    model: PyTorch model.
    dataloader: Dataloader instance for the model.
    loss_fn: PyTorch loss function to minimize.
    optimizer: PyTorch optimizer to minimize the loss function.
    device: device to compute on.

    Returns:
    Tuple of training loss and training accuracy in the form of 
    (train_loss, train_accuracy)
    """
    # set model iinto train mode
    model.train()

    # setup train loss and train accuracy
    train_loss, train_acc = 0, 0

    # loop through the data loader batches
    for batch, (X, y) in enumerate(dataloader):
        # set device
        X, y = X.to(device), y.to(device)

        # 1. forward pass
        y_pred = model(X)

        # 2. calculate cumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. loss backward
        loss.backward()

        # 4. optimizer step
        optimizer.step()

        # calculate cumulate accuracy across batches
        y_pred_class = torch.armax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.Dataloader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests PyTorch model over one epoch.
    
    Puts PyTorch model into eval mode then performs a 
    forward pass on the testing dataset.

    Args:
    model: PyTorch model.
    dataloader: Dataloader instance on which the model tests.
    loss_fn: PyTorch loss function to calculate loss.
    device: device on which to compute.

    Returns:
    Tuple of testing loss and testing acc in the form
    (test_loss, test_acc)
    """
    # set model to eval mode
    model.eval()

    # setup loss and acc
    test_loss, test_acc = 0, 0
    
    # inference context manager
    with torch.inference_mode():
        # loop through batches
        for batch, (X, y) in enumerate(dataloader):
            # to device
            X, y = X.to(device), y.to(device)
        
            # 1. forward pass
            test_pred_logits = model(X)
        
            # 2. calc and cumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
        
            # calc and cumulate acc
            test_pred_labels = test_pred_logits,argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
        
    # avg loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes model through train_step() and test_step funcs for k number of epochs,
    printing loss and acc for each epoch.

    Args:
    model: PyTorch model.
    train_dataloader: DataLoader instance on which the model trains.
    test_dataloader: DataLoader instance on which the model tests.
    optimizer: optimizer to minimize the loss.
    loss_fn: loss function to calculate loss.
    epochs: how many epochs to train.
    device: device on which to compute.

    Returns:
    dictionary of loss and acc for each epoch of both training and testing
    """
    # results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }
    
    # to device
    model.to(device)

    # Loop through training and testing steps for k epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)

        # Print results to 4 decimal places
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # filled results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

                
                
              
                
    
