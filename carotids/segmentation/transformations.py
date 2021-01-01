from numpy import asarray, where
from numpy.random import randint, uniform

from PIL import Image
from torch import Tensor
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    Resize,
)
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

    def __call__(self, img: Image, mask: Image) -> tuple:
        """Applies random vertical flip on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Image
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


class SegCrop:
    """Random crop.

    Represents random vertical crop which transforms image and segmentation mask. 
    Crops an input image so that whole object is perserved.
    """

    def __init__(self, random_t: int = 0, default_t: int = 5):
        """Initializes a random crop.

        Parameters
        ----------
        random_t : int
            TODO.
        default_t : int
            TODO.
        """
        self.random_t = random_t
        self.default_t = default_t

    def __call__(self, img: Image, mask: Image) -> tuple:
        """Applies random crop on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask: Image
            Mask to transform.

        Returns
        -------
        tuple
            Transformed image and mask.
        """
        mask_indicies = where(
            (asarray(mask) == [255, 255, 255]).any(axis=2)
        )

        x1, y1 = mask_indicies[1].min(), mask_indicies[0].min()
        x2, y2 = mask_indicies[1].max(), mask_indicies[0].max()

        width, height = img.size
        
        if self.random_t != 0:
            t = randint(self.random_t)
        
            x1, y1 = max(x1 - t, 0), max(y1 - t, 0)
            x2, y2 = min(x2 + t, width), min(y2 + t, height)
        else:
            t = self.default_t
        
            x1, y1 = max(x1 - t, 0), max(y1 - t, 0)
            x2, y2 = min(x2 + t, width), min(y2 + t, height)
        
        return img.crop((x1, y1, x2, y2)), mask.crop((x1, y1, x2, y2))


class SegCompose:
    def __init__(self, transformations: list):
        """Initializes a composition of custom transformations.

        Parameters
        ----------
        transformations: list
            List of transformations to apply.
        """
        self.transformations = transformations

    def __call__(self, img: Image, mask: Image) -> tuple:
        """Applies transformations on an image and a mask.

        Parameters
        ----------
        img : Image
            Image to be processed.
        mask : Image
            Mask to be processed.

        Returns
        -------
        tuple
            Processed image and mask.
        """
        for transformation in self.transformations:
            img, mask = transformation(img, mask)

        return img, mask
