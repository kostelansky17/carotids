from copy import deepcopy

from torch import device, set_grad_enabled
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


def train_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss: _Loss,
    optimizer: Optimizer,
    device: device,
    scheduler: _LRScheduler,
    num_epochs: int,
) -> Module:
    """Trains the model on the training data.

    Parameters
    ----------
    model : Module
        Model to train.
    train_loader : DataLoader
        Train data.
    val_loader : DataLoader
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
        Model with best validationloss during the training.
    """
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    best_val_loss = 10 ** 8
    best_model = None
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 12)

        train_epoch_loss = 0.0
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with set_grad_enabled(True):
                outputs = model(inputs)
                l = loss(outputs, labels)
                l.backward()
                optimizer.step()

                train_epoch_loss += l.item() * inputs.size(0)

        scheduler.step(train_epoch_loss)

        val_epoch_loss = 0.0
        model.eval()

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with set_grad_enabled(False):
                outputs = model(inputs)
                l = loss(outputs, labels)
                val_epoch_loss += l.item() * inputs.size(0)

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = deepcopy(model.state_dict())

        print(
            f"Train loss: {train_epoch_loss/train_size}, Val. loss: {val_epoch_loss/val_size}"
        )

    model.load_state_dict(best_model)
    return model
