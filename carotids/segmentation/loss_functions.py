from torch import cosh, log, Tensor
from torch.nn import Module, Softmax


class DiceLoss(Module):
    """Dice loss used for the segmentation tasks."""

    def __init__(self):
        """Initializes a dice loss function."""

        super(DiceLoss, self).__init__()
        self.soft_max = Softmax(dim=1)

    def forward(self, inputs: Tensor, targets: Tensor):
        """Computes dice loss between the input and the target values.

        Parameters
        ----------
        inputs : Tensor
            Values predicted by the model.
        targets : Tensor
            The target values.
        
        Returns
        -------
        Tensor
            The dice loss between the input and the target values.
        """

        inputs = self.soft_max(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection) / (inputs.sum() + targets.sum())

        return 1 - dice


class LogCoshDiceLoss(Module):
    def __init__(self):
        """Initializes a log-cosh dice loss function."""

        super(LogCoshDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        """Computes log-cosh dice loss between the input and the target values.

        Parameters
        ----------
        inputs : Tensor
            Values predicted by the model.
        targets : Tensor
            The target values.
        
        Returns
        -------
        Tensor
            The log-cosh dice loss between the input and the target values.
        """

        dice_loss = self.dice_loss(inputs, targets)

        return log(cosh(dice_loss))