from numpy import mean
from torch import device, logical_and, logical_or, no_grad, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from carotids.segmentation.loss_functions import DiceLoss, LogCoshDiceLoss


@no_grad()
def dataset_dice_loss(dataset: Dataset, model: Module, device: device) -> Tensor:
    """Computes dice loss between the input and the target values.

    Parameters
    ----------
    dataset : Dataset
    
    model : Module
    
    device : device
    
    
    Returns
    -------
    Tensor
        .
    """
    dice_loss = DiceLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sum_dice_loss = 0.0
    for data, label in dataloader:
        data.to(device)
        label.to(device)

        prediction = model(data)

        sum_dice_loss += dice_loss(prediction, label)

    return sum_dice_loss / len(dataset)


@no_grad()
def dataset_logcosh_dice_loss(
    dataset: Dataset, model: Module, device: device
) -> tensor:
    lc_dice_loss = LogCoshDiceLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sum_lc_dice_loss = 0.0
    for data, label in dataloader:
        data.to(device)
        label.to(device)

        prediction = model(data)

        sum_lc_dice_loss += lc_dice_loss(prediction, label)

    return sum_lc_dice_loss / len(dataset)


@no_grad()
def dataset_classes_iou(
    dataset: Dataset, model: Module, n_classes: int, device: device
) -> list:
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    classes_iou = {i: [] for i in range(n_classes)}
    for data, label in dataloader:
        data.to(device)
        label.to(device)

        prediction = model(data)

        prediction = prediction.squeeze().argmax(0)
        label = label.squeeze().argmax(0)

        for i in range(n_classes):
            classes_iou[i].append(
                logical_and(prediction == i, label == i).sum()
                / logical_or(prediction == i, label == i).sum()
            )

    return [mean(classes_iou[i]) for i in range(n_classes)]
