from torch import device
from torch.nn import Module
from torch.utils.data import DataLoader

from carotids.localization.utils import MetricLogger, reduce_dict


def eval_one_epoch(
    model: Module, data_loader: DataLoader, device: device, print_freq: int
):
    """Evaluates Faster R-CNN on the dataloader.

    Parameters
    ----------
    model : Module
        Model to train.
    data_loader : DataLoader
        Train data.
    device : device
        Device on which is the model.
    print_freq : int
        The printing frequency during the training.

    Returns
    -------
    MetricLogger:
        Statistics about the performance.
    """
    metric_logger = MetricLogger(delimiter="  ")

    for images, targets in metric_logger.log_every(data_loader, print_freq):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

    return metric_logger
