from torch import cat, tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    MaxPool2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
)


class Unet(Module):
    """U-net model.

    Represents implementation of the U-net model for image segmentation.
    """

    def __init__(self, number_of_classes: int) -> None:
        """Initializes a U-net model.

        Parameters
        ----------
        number_of_classes : int
            Number of classes in segmentation
        """
        super(Unet, self).__init__()

        self.L0 = LeftBlock(3, 64)
        self.L1 = LeftBlock(64, 128)
        self.L2 = LeftBlock(128, 256)
        self.L3 = LeftBlock(256, 512)
        self.L4 = LeftBlock(512, 1024)

        self.R3 = RightBlock(1024, 512)
        self.R2 = RightBlock(512, 256)
        self.R1 = RightBlock(256, 128)
        self.R0 = RightBlock(128, 64)

        self.last_layer = Conv2d(64, number_of_classes, kernel_size=1)

    def forward(self, input: tensor) -> tensor:
        """Applies model to an input.

        Parameters
        ----------
        input : tensor
            Image to use as an input.

        Returns
        -------
        tensor:
            Segmentation of an input.
        """
        input, l0 = self.L0(input)
        input, l1 = self.L1(input)
        input, l2 = self.L2(input)
        input, l3 = self.L3(input)
        _, l4 = self.L4(input)

        input = self.R3(l4, l3)
        input = self.R2(input, l2)
        input = self.R1(input, l1)
        input = self.R0(input, l0)

        return self.last_layer(input)


class ConvBlock(Module):
    """Block of convolutional layers.

    Combines two convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3),
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.25,
    ):
        """Initializes a Block of convolutional layers.

        Parameters
        ----------
        in_channels : int
            Number of input channels to a convolutional block.
        out_channels : int
            Number of output channels to a convolutional block.
        kernel_size : tuple
            Size of a kernel in a convolutional layer.
        stride : int
            Stride used in a convolutional layer.
        padding : int
            Number of padded pixels in a convolutional layer.
        """
        super(ConvBlock, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            PReLU(),
            Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            BatchNorm2d(out_channels),
            PReLU(),
        )

    def forward(self, input: tensor) -> tensor:
        """Applies a block of convolutional layers to an input.

        Parameters
        ----------
        input : tensor
            An input tensor.

        Returns
        -------
        tensor
            A tensor after applying convolutional layers.
        """
        input = self.layers(input)

        return input


class LeftBlock(Module):
    """Left block of U-net model.

    Combines a block of two convolutional layers and maxpooling layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3),
        stride: int = 1,
        padding: int = 1,
        pool_kernel_size: int = 2,
    ):
        """Initializes a Left Block.

        Parameters
        ----------
        in_channels : int
            Number of input channels to a convolutional block.
        out_channels : int
            Number of output channels to a convolutional block.
        kernel_size : tuple
            Size of a kernel in a convolutional layer.
        stride : int
            Stride used in a convolutional layer.
        padding : int
            Number of padded pixels in a convolutional layer.
        pool_kernel_size : int
            Size of a kernel in a maxpooling layer.
        """
        super(LeftBlock, self).__init__()

        self.conv_layers = ConvBlock(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.pool_layer = MaxPool2d(pool_kernel_size)

    def forward(self, input: tensor) -> tuple:
        """Applies a block of layers to an input.

        Parameters
        ----------
        input : tensor
            Image to use as an input.

        Returns
        -------
        tuple
            A tensor processed by a block of convolutional layers (input to right block)
            and a tensor process by a block of convolutional layers and maxpooling layer
            (input to the next Left Block).
        """
        output = self.conv_layers(input)
        return self.pool_layer(output), output


class RightBlock(Module):
    """Right block of U-net model.

    Combines Upsampling and a block of two convolutional layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3),
        stride: int = 1,
        padding: int = 1,
        scale_factor: int = 2,
    ):
        """Initializes a Left Block.

        Parameters
        ----------
        in_channels : int
            Number of input channels to a convolutional block.
        out_channels : int
            Number of output channels to a convolutional block.
        kernel_size : tuple
            Size of a kernel in a convolutional layer.
        stride : int
            Stride used in a convolutional layer.
        padding : int
            Number of padded pixels in a convolutional layer.
        scale_factor : int
            Scale factor of an upsampling.
        """
        super(RightBlock, self).__init__()

        self.up_layer = Sequential(
            Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True),
            ConvBlock(in_channels, out_channels, kernel_size, stride, padding),
        )
        self.conv_layers = ConvBlock(
            in_channels, out_channels, kernel_size, stride, padding
        )

    def forward(self, down_input: tensor, left_input: tensor) -> tensor:
        """Applies a block of layers to an input.

        Parameters
        ----------
        down_input : tensor
            A tensor from a right block of a model.
        left_input : tensor
            A tensor from a left block of a model.
        Returns
        -------
        tensor
            Tensor created by applying block of layers to inputs.
        """
        down_input = self.up_layer(down_input)

        input = cat((left_input, down_input), dim=1)
        output = self.conv_layers(input)

        return output
