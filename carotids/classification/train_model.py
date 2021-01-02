from copy import deepcopy
from typing import Union

from torch import device, set_grad_enabled
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from carotids.metrics import accuracy_torch, evaluate_classification_model


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss: _Loss,
    optimizer: Optimizer,
    device: device,
    scheduler: Union[None, _LRScheduler] = None,
    num_epochs: int = 75,
) -> tuple:
    """Trains the model on the training data.

    Parameters
    ----------
    model : Module
        Model to train.
    train_data : DataLoader
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

    train_size = len(train_loader.dataset)

    best_model = deepcopy(model.state_dict())
    best_loss = 10 ** 8
    best_accuracy = 0.0

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
            with set_grad_enabled(True):
                outputs = model(inputs)

                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                train_epoch_loss += l.item() * inputs.size(0)
                train_epoch_acc += accuracy_torch(outputs, labels) * inputs.size(0)

        if scheduler:
            scheduler.step(train_epoch_loss / train_size)

        val_epoch_loss, val_epoch_acc = evaluate_classification_model(
            model, val_loader, loss, device
        )

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_accuracy = val_epoch_acc
            best_model = deepcopy(model.state_dict())

        losses["train"].append(train_epoch_loss / train_size)
        losses["val"].append(val_epoch_loss)

        accuracies["train"].append(train_epoch_acc / train_size)
        accuracies["val"].append(val_epoch_acc)

        print(
            f"Train loss: {train_epoch_loss / train_size}, Train Accuracy: {train_epoch_acc / train_size}"
        )
        print(f"Val. loss: {val_epoch_loss}, Val. Accuracy: {val_epoch_acc}")

    print("-" * 12)
    print(f"Best val. loss: {best_loss}, Accuracy: {best_accuracy}")

    model.load_state_dict(best_model)
    return model, losses, accuracies
