from numpy import array, asarray
from numpy.random import randint, uniform
from PIL import Image
from torch import Tensor
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Resize
from torchvision.transforms.functional import crop

HORIZONTAL_FLIP = RandomHorizontalFlip(1.0)
VERTICAL_FLIP = RandomVerticalFlip(1.0)


class SegRandomHorizontalFlip:
    """Random horizontal flip.

    Represents random horizontal flip which transforms image and segmentation mask.
    """

    def __init__(self, p: float = 0.5):
        """Initializes a random horizontal flip.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        """
        self.p = p

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies random horizontal flip on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to transform.

        Returns
        -------
        tuple
            Transformed image and mask.
        """
        if uniform() <= self.p:
            img = HORIZONTAL_FLIP(img)
            mask = HORIZONTAL_FLIP(mask)

        return img, mask


class SegRandomVerticalFlip:
    """Random vertical flip.

    Represents random vertical flip which transforms image and segmentation mask.
    """

    def __init__(self, p: float = 0.5):
        """Initializes a random vertical flip.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        """
        self.p = p

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies random vertical flip on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to transform.

        Returns
        -------
        tuple
            Transformed image and mask.
        """
        if uniform() <= self.p:
            img = VERTICAL_FLIP(img)
            mask = VERTICAL_FLIP(mask)

        return img, mask


class LocCrop:
    """Random crop.

    Represents random vertical crop which transforms image and segmentation mask. 
    Crops an input image so that whole object is perserved.
    """

    def __init__(self, p: float = 0.5):
        """Initializes a random crop.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        """
        self.p = p

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies random crop on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask: Tensor
            Mask to transform.

        Returns
        -------
        tuple
            Transformed image and mask.
        """
        if uniform() <= self.p:
            img = crop(img, crop_y0, crop_x0, crop_h, crop_w)
        else:
            pass

        return img, mask


class SegCompose:
    def __init__(self, transformations: list):
        """Initializes a composition of custom transformations.

        Parameters
        ----------
        transformations: list
            List of transformations to apply.
        """
        self.transformations = transformations

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies transformations on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to be processed.
        mask : Tensor
            Mask to be processed.
        
        Returns
        -------
        tuple
            Processed image and mask.
        """
        for transformation in self.transformations:
            img, mask = transformation(img, mask)

        return img, mask
