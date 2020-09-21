from copy import deepcopy
from typing import Union

from torch import device, Optimizer
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from carotids.metrics import accuracy_torch
from carotids.utils import train_val_split


def train_model(
    model: Module,
    train_data: Dataset,
    loss: _Loss,
    optimizer: Optimizer,
    device: device,
    scheduler: Union[None, _LRScheduler] = None,
    val_split: float = 0.2,
    num_epochs: int = 75,
) -> tuple:
    """Trains the model on the training data.

    Parameters
    ----------
    model : Module
        Model to train.
    train_data : Dataset
        Train data.
    loss : _Loss
        Loss function.
    optimizer : Optimizer
        Selected optimizer which updates weights of the model
    device : device
        Device on which is the model.
    scheduler : Union[None, _LRScheduler]
        Selected scheduler of the learning rate.
    val_split : float
        Ratio of the train-validation split.
    num_epochs : int
        Number of training epochs.

    Returns
    -------
    tuple
        Model with best loss and the loss and accuracy metrics from observed
        during the training.
    """
    losses = {"train": [], "val": []}
    accuracies = {"train": [], "val": []}

    train_loader, train_size, val_loader, val_size = train_val_split(
        train_data, val_split
    )

    best_model = copy.deepcopy(model.state_dict())
    best_loss = 10 ** 8

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)

        train_epoch_loss = 0.0
        train_epoch_acc = 0
        for inputs, labels in train_loader:
            model.train()

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)

                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                if scheduler:
                    scheduler.step()

                train_epoch_loss += l.item() * inputs.size(0)
                train_epoch_acc += accuracy_torch(outputs, labels) * inputs.size(0)

        val_epoch_loss = 0.0
        val_epoch_acc = 0
        for inputs, labels in val_loader:
            model.eval()

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            l = loss(outputs, labels)

            val_epoch_loss += l.item() * inputs.size(0)
            val_epoch_acc += accuracy_torch(outputs, labels) * inputs.size(0)

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_model = deepcopy(model.state_dict())

        losses["train"].append(train_epoch_loss / train_size)
        losses["val"].append(val_epoch_loss / val_size)

        accuracies["train"].append(train_epoch_acc / train_size)
        accuracies["val"].append(val_epoch_acc / val_size)

        print(
            f"Train loss: {train_epoch_loss / train_size}, Train Accuracy: {train_epoch_acc / train_size}"
        )
        print(
            f"Val. loss: {val_epoch_loss / val_size}, Val. Accuracy: {val_epoch_acc / val_size}"
        )

    print("-" * 12)
    print(f"Best val. loss: {best_loss/val_size}")

    model.load_state_dict(best_model)
    return model, losses, accuracies
