from torch import cat, tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Dropout2d,
    MaxPool2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
)


class Unet(Module):
    """U-net model.

    Represents implementation of U-net model for image segmentation.
    """

    def __init__(
        self,
        number_of_classes: int,
        conv_kernel_size: tuple = (3, 3),
        conv_stride: int = 1,
        conv_padding: int = 1,
        pool_kernel_size: int = 2,
        up_scale_factor: int = 2,
        dropout_p: float = 0.25,
    ) -> None:
        """Initializes U-net model.

        Parameters
        ----------
        number_of_classes : int
            Number of classes in segmentation
        kernel_size : tuple
            Size of a kernel in a convolutional layer.
        stride : int
            Stride used in a convolutional layer.
        padding : int
            Number of padded pixels in a convolutional layer.
        pool_kernel_size : int
            Size of a kernel in a maxpooling layer.
        up_scale_factor : int
            Scale factor of an upsampling.
        dropout_p : float
            Probability of an element to be zeroed.
        """
        super(Unet, self).__init__()

        self.L0 = LeftBlock(
            3,
            64,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L1 = LeftBlock(
            64,
            128,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L2 = LeftBlock(
            128,
            256,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L3 = LeftBlock(
            256,
            512,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L4 = LeftBlock(
            512,
            1024,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )

        self.R3 = RightBlock(
            1024,
            512,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )
        self.R2 = RightBlock(
            512,
            256,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )
        self.R1 = RightBlock(
            256,
            128,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )
        self.R0 = RightBlock(
            128,
            64,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )

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


class BigUnet(Module):
    """Big U-net model.

    Represents implementation of Big U-net model for image segmentation. This
    architecture adds additional layer in both arms.
    """

    def __init__(
        self,
        number_of_classes: int,
        conv_kernel_size: tuple = (3, 3),
        conv_stride: int = 1,
        conv_padding: int = 1,
        pool_kernel_size: int = 2,
        up_scale_factor: int = 2,
        dropout_p: float = 0.25,
        n_fillter_exponent: int = 8,
    ) -> None:
        """Initializes U-net model.

        Parameters
        ----------
        number_of_classes : int
            Number of classes in segmentation
        kernel_size : tuple
            Size of a kernel in a convolutional layer.
        stride : int
            Stride used in a convolutional layer.
        padding : int
            Number of padded pixels in a convolutional layer.
        pool_kernel_size : int
            Size of a kernel in a maxpooling layer.
        up_scale_factor : int
            Scale factor of an upsampling.
        dropout_p : float
            Probability of an element to be zeroed.
        n_fillter_exponent : int
            Exponent of 2 defining the begining number of convolutional filters.
        """
        super(BigUnet, self).__init__()

        self.L0 = LeftBlock(
            3,
            2 ** n_fillter_exponent,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L1 = LeftBlock(
            2 ** n_fillter_exponent,
            2 ** (n_fillter_exponent + 1),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L2 = LeftBlock(
            2 ** (n_fillter_exponent + 1),
            2 ** (n_fillter_exponent + 2),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L3 = LeftBlock(
            2 ** (n_fillter_exponent + 2),
            2 ** (n_fillter_exponent + 3),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L4 = LeftBlock(
            2 ** (n_fillter_exponent + 3),
            2 ** (n_fillter_exponent + 4),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )
        self.L5 = LeftBlock(
            2 ** (n_fillter_exponent + 4),
            2 ** (n_fillter_exponent + 5),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            pool_kernel_size,
            dropout_p,
        )

        self.R4 = RightBlock(
            2 ** (n_fillter_exponent + 5),
            2 ** (n_fillter_exponent + 4),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )
        self.R3 = RightBlock(
            2 ** (n_fillter_exponent + 4),
            2 ** (n_fillter_exponent + 3),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )
        self.R2 = RightBlock(
            2 ** (n_fillter_exponent + 3),
            2 ** (n_fillter_exponent + 2),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )
        self.R1 = RightBlock(
            2 ** (n_fillter_exponent + 2),
            2 ** (n_fillter_exponent + 1),
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )
        self.R0 = RightBlock(
            2 ** (n_fillter_exponent + 1),
            2 ** n_fillter_exponent,
            conv_kernel_size,
            conv_stride,
            conv_padding,
            up_scale_factor,
            dropout_p,
        )

        self.last_layer = Conv2d(
            2 ** n_fillter_exponent, number_of_classes, kernel_size=1
        )

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
        input, l4 = self.L4(input)
        _, l5 = self.L5(input)

        input = self.R4(l5, l4)
        input = self.R3(l4, l3)
        input = self.R2(input, l2)
        input = self.R1(input, l1)
        input = self.R0(input, l0)

        return self.last_layer(input)




class UnetDVCFS(Module):
    """U-net model wth depth-wise convolutional filter size.

    Represents implementation of U-net model for image segmentation.
    """

    def __init__(
        self,
        number_of_classes: int,
        pool_kernel_size: int = 2,
        up_scale_factor: int = 2,
        dropout_p: float = 0.25,
    ) -> None:
        """Initializes U-net model.

        Parameters
        ----------
        number_of_classes : int
            Number of classes in segmentation
        kernel_size : tuple
            Size of a kernel in a convolutional layer.
        stride : int
            Stride used in a convolutional layer.
        padding : int
            Number of padded pixels in a convolutional layer.
        pool_kernel_size : int
            Size of a kernel in a maxpooling layer.
        up_scale_factor : int
            Scale factor of an upsampling.
        dropout_p : float
            Probability of an element to be zeroed.
        """
        super(UnetDVCFS, self).__init__()

        self.L0 = LeftBlock(
            3,
            64,
            11,
            1,
            5,
            pool_kernel_size,
            dropout_p,
        )
        self.L1 = LeftBlock(
            64,
            128,
            9,
            1,
            4,
            pool_kernel_size,
            dropout_p,
        )
        self.L2 = LeftBlock(
            128,
            256,
            7,
            1,
            3,
            pool_kernel_size,
            dropout_p,
        )
        self.L3 = LeftBlock(
            256,
            512,
            5,
            1,
            2,
            pool_kernel_size,
            dropout_p,
        )
        self.L4 = LeftBlock(
            512,
            1024,
            3,
            1,
            1,
            pool_kernel_size,
            dropout_p,
        )

        self.R3 = RightBlock(
            1024,
            512,
            5,
            1,
            2,
            up_scale_factor,
            dropout_p,
        )
        self.R2 = RightBlock(
            512,
            256,
            7,
            1,
            3,
            up_scale_factor,
            dropout_p,
        )
        self.R1 = RightBlock(
            256,
            128,
            9,
            1,
            4,
            up_scale_factor,
            dropout_p,
        )
        self.R0 = RightBlock(
            128,
            64,
            11,
            1,
            5,
            up_scale_factor,
            dropout_p,
        )

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
        kernel_size: tuple,
        stride: int,
        padding: int,
        dropout_p: float,
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
        dropout_p : float
            Probability of an element to be zeroed.
        """
        super(ConvBlock, self).__init__()

        self.layers = Sequential(
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            PReLU(),
            BatchNorm2d(out_channels),
            Dropout2d(dropout_p),
            Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            PReLU(),
            BatchNorm2d(out_channels),
            Dropout2d(dropout_p),
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
        kernel_size: tuple,
        stride: int,
        padding: int,
        pool_kernel_size: int,
        dropout_p: float,
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
        dropout_p : float
            Probability of an element to be zeroed.
        """
        super(LeftBlock, self).__init__()

        self.conv_layers = ConvBlock(
            in_channels, out_channels, kernel_size, stride, padding, dropout_p
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
        kernel_size: tuple,
        stride: int,
        padding: int,
        up_scale_factor: int,
        dropout_p: float,
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
        up_scale_factor : int
            Scale factor of an upsampling.
        dropout_p : float
            Probability of an element to be zeroed.
        """
        super(RightBlock, self).__init__()

        self.up_layer = Sequential(
            Upsample(scale_factor=up_scale_factor, mode="bilinear", align_corners=True),
            Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        )
        self.conv_layers = ConvBlock(
            in_channels, out_channels, kernel_size, stride, padding, dropout_p
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

