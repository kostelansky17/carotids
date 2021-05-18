from numpy import asarray, where
from numpy.random import normal, randint, uniform

from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_gamma,
    crop,
    hflip,
    rotate,
    vflip,
)


class SegRandomRotation:
    """Random rotation.

    Represents random rotation which transforms image and segmentation mask.
    """

    def __init__(self, p: float = 0.5, min_angle: int = -10, max_angle: int = 10):
        """Initializes a random rotation.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        min_angle : int
            Minimal angle bound for rotation (inclusive).
        max_angle : int
            Max angle bound for rotation (exclusive).
        """
        self.p = p
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies random rotation on an image and a mask.

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
            rotation_angle = randint(self.min_angle, self.max_angle)
            return rotate(img, rotation_angle), rotate(mask, rotation_angle)

        return img, mask


class SegRandomContrast:
    """Random contrast.

    Represents random contrast which transforms image.
    """

    def __init__(self, p: float = 0.5, mean: float = 1.0, std: float = 0.025):
        """Initializes a random contrast.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        mean : float
            Mean of the contrast factor distribution for transformation.
        std : float
            Standard deviance of the contrast factor distribution for
            transformation.
        """
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies contrast on an image.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to pass.

        Returns
        -------
        tuple
            Transformed image and original mask.
        """
        if uniform() <= self.p:
            contrast_factor = max(0, normal(self.mean, self.std))
            return adjust_contrast(img, contrast_factor), mask

        return img, mask


class SegRandomGammaCorrection:
    """Random gamma correction.

    Represents random gamma correction which transforms image.
    """

    def __init__(self, p: float = 0.5, mean: float = 1.0, std: float = 0.025):
        """Initializes a random gamma correction.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        mean : float
            Mean of the gamma correction distribution for transformation.
        std : float
            Standard deviance of the gamma correction distribution for
            transformation.
        """
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies gamma correction on an image.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to pass.

        Returns
        -------
        tuple
            Transformed image and original mask.
        """
        if uniform() <= self.p:
            gamma = max(0, normal(self.mean, self.std))
            return adjust_gamma(img, gamma), mask

        return img, mask


class SegRandomBrightness:
    """Random brightness.

    Represents random brightness which transforms image.
    """

    def __init__(self, p: float = 0.5, mean: float = 1.0, std: float = 0.025):
        """Initializes a random brightness.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        mean : float
            Mean of the brightness factor distribution for transformation.
        std : float
            Standard deviance of the brightness factor distribution for
            transformation.
        """
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img: Image, mask: Tensor) -> tuple:
        """Applies brightness on an image.

        Parameters
        ----------
        img : Image
            Image to transform.
        mask : Tensor
            Mask to pass.

        Returns
        -------
        tuple
            Transformed image and original mask.
        """
        if uniform() <= self.p:
            brightness_factor = max(0, normal(self.mean, self.std))
            return adjust_brightness(img, brightness_factor), mask

        return img, mask


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
            return hflip(img), hflip(mask)

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
            return vflip(img), vflip(mask)

        return img, mask


class SegCrop:
    """Random crop.

    Represents random crop which transforms image and segmentation mask.
    Crops an input image so that whole object is perserved. This is done
    by adding a random number of pixels to each side. Also has a default
    setting that adds fixed number of pixels to each side.
    """

    def __init__(self, random_t: int = 0, default_t: int = 5):
        """Initializes a random crop.

        Parameters
        ----------
        random_t : int
            Upper bound for number of pixels added.
        default_t : int
            Fixed number of pixels added to a side.
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
        mask_indices = where((asarray(mask) == [255, 255, 255]).any(axis=2))

        x1, y1 = mask_indices[1].min(), mask_indices[0].min()
        x2, y2 = mask_indices[1].max(), mask_indices[0].max()

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
    def __init__(self, transformations: list = []):
        """Initializes a composition of custom segmentation transformations.

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
