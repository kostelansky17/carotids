from torch import tensor
from torch.nn import Module


class Unet(Module):
    """U-net model.

    Represents implementation of the U-net model for image segmentation.
    """

    def __init__(self) -> None:
        """Initializes a U-net model.

        Parameters
        ----------
        """
        pass

    def forward(self, img: tensor) -> tensor:
        """Applies model to an input.

        Parameters
        ----------
        img : tensor
            Image to use as an input.

        Returns
        -------
        tensor:
            Segmentation of an input.
        """
        pass


class Forward:
    def __init__(kernel_size: tuple = (3, 3), stride: int = 1, filters: int = 64):
        self.kernel_size = kernel_size
        self.stride = stride
        self.filters = filters
