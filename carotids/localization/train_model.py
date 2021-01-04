from torch import device
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from carotids.localization.utils import MetricLogger, reduce_dict, SmoothedValue


def train_one_epoch(
    model: Module,
    optimizer: Optimizer,
    data_loader: DataLoader,
    device: device,
    epoch: int,
    print_freq: int,
):
    """Trains Faster R-CNN for one epoch on the data loader.

    Parameters
    ----------
    model : Module
        Model to train.
    optimizer : Optimizer
        Selected optimizer which updates weights of the model
    data_loader : DataLoader
        Train data.
    device : device
        Device on which is the model.
    epoch : int
        The number of the training epoch.
    print_freq : int
        The printing frequency during the training.

    Returns
    -------
    MetricLogger:
        Statistics about the training epoch.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger
