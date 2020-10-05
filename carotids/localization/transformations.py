from numpy import array, asarray
from numpy.random import uniform
from PIL import Image
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

HORIZONTAL_FLIP = RandomHorizontalFlip(1.0)
VERTICAL_FLIP = RandomVerticalFlip(1.0)


class LocRandomHorizontalFlip:
    """Random horizontal flip.

    Represents random horizontal flip which transforms image and bounding box.
    """
    def __init__(self, p:float=0.5):
        """Initializes a random horizontal flip.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        """
        self.p = p

    def __call__(self, img: Image, bounding_box: array) -> tuple:
        """Applies random horizontal flip on an image and a bounding box.

        Parameters
        ----------
        img : Image
            Image to transform.
        bounding_box : array
            Bounding box to transform.

        Returns
        -------
        tuple
            Transformed image and bounding box.
        """
        w, h = img.size

        if uniform() < self.p:
            img = HORIZONTAL_FLIP(img)

            x0 = w - bounding_box[2]
            y0 = bounding_box[1]
            x1 = w - bounding_box[0]
            y1 = bounding_box[3]

            bounding_box = asarray([x0, y0, x1, y1])

        return img, bounding_box


class LocRandomVerticalFlip:
    """Random vertical flip.

    Represents random vertical flip which transforms image and bounding box.
    """
    def __init__(self, p: float=0.5):
        """Initializes a random vertical flip.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        """
        self.p = p

    def __call__(self, img: Image, bounding_box: array) -> tuple:
        """Applies random vertical flip on an image and a bounding box.

        Parameters
        ----------
        img : Image
            Image to transform.
        bounding_box : array
            Bounding box to transform.

        Returns
        -------
        tuple
            Transformed image and bounding box.
        """
        w, h = img.size

        if uniform() < self.p:
            img = VERTICAL_FLIP(img)

            x0 = bounding_box[0]
            y0 = h - bounding_box[3]
            x1 = bounding_box[2]
            y1 = h - bounding_box[1]

            bounding_box = asarray([x0, y0, x1, y1])

        return img, bounding_box


def transform_item(
    img: Image, bounding_box: tuple, transformations: list
) -> tuple:
    """Applies transformations on an image and a bounding box.

    Parameters
    ----------
    img : Image
        Image to be processed.
    bounding_box : tuple
        Bounding box to be processed.
    transformations : list
        Transformations to apply.   
    
    Returns
    -------
    tuple
        Processed image and bounding box.
    """
    for transformation in transformations:
        img, bounding_box = transformation(img, bounding_box)

    return img, bounding_box
