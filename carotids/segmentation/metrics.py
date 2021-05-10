from numpy import mean, prod
from torch import device, logical_and, logical_or, no_grad, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from carotids.segmentation.loss_functions import DiceLoss, LogCoshDiceLoss


@no_grad()
def dataset_dice_loss(dataset: Dataset, model: Module, device: device) -> Tensor:
    """Computes mean dice loss between the values predicted by the model and the
    ground truth.

    Parameters
    ----------
    dataset : Dataset
        Dataset to compute the mean dice loss on.
    model : Module
        The model to evaluate.
    device : device
        The device which is used for computation.

    Returns
    -------
    Tensor
        Mean dice loss.
    """
    model.eval()
    dice_loss = DiceLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sum_dice_loss = 0.0
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)

        prediction = model(data)

        sum_dice_loss += dice_loss(prediction, label)

    return sum_dice_loss / len(dataset)


@no_grad()
def dataset_logcosh_dice_loss(
    dataset: Dataset, model: Module, device: device
) -> Tensor:
    """Computes mean log-cosh dice loss between the values predicted by the
    model and the ground truth.

    Parameters
    ----------
    dataset : Dataset
        Dataset to compute the mean log-cosh dice loss on.
    model : Module
        The model to evaluate.
    device : device
        The device which is used for computation.

    Returns
    -------
    Tensor
        Mean log-cosh dice loss.
    """
    model.eval()
    lc_dice_loss = LogCoshDiceLoss()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    sum_lc_dice_loss = 0.0
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)

        prediction = model(data)

        sum_lc_dice_loss += lc_dice_loss(prediction, label)

    return sum_lc_dice_loss / len(dataset)


@no_grad()
def dataset_classes_iou(
    dataset: Dataset, model: Module, n_classes: int, device: device
) -> list:
    """Computes mean IoU for every class between the values predicted by the
    model and the ground truth.

    Parameters
    ----------
    dataset : Dataset
        Dataset to compute the mean IoU for every class.
    model : Module
        The model to evaluate.
    device : device
        The device which is used for computation.

    Returns
    -------
    list
        Mean IoU for every class.
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    classes_iou = {i: [] for i in range(n_classes)}
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)

        prediction = model(data)

        prediction = prediction.squeeze().argmax(0)
        label = label.squeeze().argmax(0)

        for i in range(n_classes):
            classes_iou[i].append(
                logical_and(prediction == i, label == i).sum()
                / logical_or(prediction == i, label == i).sum()
            )

    return [mean(classes_iou[i]) for i in range(n_classes)]


class SegAccuracy(Module):
    """Accuracy used for the segmentation tasks."""

    def __init__(self, size: tuple) -> None:
        """Initializes the Accuracy module.

        Parameters
        ----------
        size : tuple
            The size of the network's input.
        """
        super(SegAccuracy, self).__init__()
        self.size = prod(size)

    def forward(self, targets: Tensor, outputs: Tensor):
        """.

        Parameters
        ----------
        targets : Tensor
            The ground truth annotation.
        outputs : Tensor
            The classes predicted by the model.
        """
        outputs = outputs.argmax(dim=1)
        targets = targets.argmax(dim=1)
        
        return (outputs == targets).sum() / self.size
