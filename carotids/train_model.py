import copy

import torch
from torch.utils.data import DataLoader

from carotids.metrics import accuracy_torch
from carotids.utils import train_val_split


def train_model(
    model,
    train_data,
    test_data,
    loss,
    optimizer,
    device,
    scheduler=None,
    val_split=0.1,
    num_epochs=25,
):
    losses = {"train": [], "val": []}
    accuracies = {"train": [], "val": []}

    #train_loader, train_size, val_loader, val_size = train_val_split(
    #    train_data, val_split
    #)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    train_size = len(train_loader)
    val_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    val_size = len(val_loader)

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
            best_model = copy.deepcopy(model.state_dict())

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
