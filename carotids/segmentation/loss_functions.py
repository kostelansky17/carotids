from torch import cosh, log, ones, Tensor
from torch.nn import Module, Softmax
from torch.nn.functional import softmax

class DiceLoss(Module):
    """Dice loss used for the segmentation tasks."""

    def __init__(self, weights: list = []):
        """Initializes a dice loss function.

        Parameters
        ----------
        weights : list
            Rescaling weights given to each class.
        """
        super(DiceLoss, self).__init__()
        self.soft_max = Softmax(dim=1)
        self.weights = weights

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Computes dice loss between the input and the target values.

        Parameters
        ----------
        outputs : Tensor
            Values predicted by the model.
        targets : Tensor
            The target values.

        Returns
        -------
        Tensor
            The dice loss between the input and the target values.
        """

        outputs = self.soft_max(outputs.float())

        for i, w in enumerate(self.weights):
            targets[:, i, :, :] = targets[:, i, :, :] * w

        outputs = outputs.view(-1)
        targets = targets.view(-1)

        intersection = (outputs * targets).sum()
        dice = (2.0 * intersection) / (outputs.sum() + targets.sum())

        return 1 - dice


class LogCoshDiceLoss(Module):
    def __init__(self, weights: list = []):
        """Initializes a log-cosh dice loss function.

        weights : list
            Rescaling weights given to each class.
        """
        super(LogCoshDiceLoss, self).__init__()
        self.dice_loss = DiceLoss(weights)

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Computes log-cosh dice loss between the input and the target values.

        Parameters
        ----------
        outputs : Tensor
            Values predicted by the model.
        targets : Tensor
            The target values.

        Returns
        -------
        Tensor
            The log-cosh dice loss between the input and the target values.
        """

        dice_loss = self.dice_loss(outputs, targets)

        return log(cosh(dice_loss))


def logcosh_dice_loss(outputs: Tensor, targets: Tensor, weights: list = []) -> Tensor:
    outputs = softmax(outputs.float(), dim=1)

    for i, w in enumerate(weights):
        targets[:, i, :, :] = targets[:, i, :, :] * w

    outputs = outputs.view(-1)
    targets = targets.view(-1)

    intersection = (outputs * targets).sum()
    dice = (2.0 * intersection) / (outputs.sum() + targets.sum())

    return log(cosh(1 - dice))