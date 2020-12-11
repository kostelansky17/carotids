from numpy import array, asarray
from numpy.random import randint, uniform
from PIL import Image
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, Resize
from torchvision.transforms.functional import crop

HORIZONTAL_FLIP = RandomHorizontalFlip(1.0)
VERTICAL_FLIP = RandomVerticalFlip(1.0)


class LocRandomHorizontalFlip:
    """Random horizontal flip.

    Represents random horizontal flip which transforms image and bounding box.
    """

    def __init__(self, p: float = 0.5):
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
        if uniform() <= self.p:
            w, h = img.size
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

    def __init__(self, p: float = 0.5):
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
        if uniform() <= self.p:
            w, h = img.size
            img = VERTICAL_FLIP(img)

            x0 = bounding_box[0]
            y0 = h - bounding_box[3]
            x1 = bounding_box[2]
            y1 = h - bounding_box[1]

            bounding_box = asarray([x0, y0, x1, y1])

        return img, bounding_box


class LocCrop:
    """Random crop.

    Represents random vertical crop which transforms image and bounding box. 
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

    def __call__(self, img: Image, bounding_box: array) -> tuple:
        """Applies random crop on an image and a bounding box.

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
        x0, y0, x1, y1 = bounding_box

        if uniform() <= self.p:
            w, h = img.size
            if bounding_box[0] > 0 and bounding_box[2] < w:
                crop_x0 = randint(0, bounding_box[0])
                crop_w = randint(bounding_box[2], w) - crop_x0
                
                img = crop(
                    img,
                    0,
                    crop_x0,
                    h,
                    crop_w
                )

                x0 = bounding_box[0] - crop_x0
                x1 = bounding_box[2] - crop_x0
            

            w, h = img.size
            if bounding_box[1] > 0 and bounding_box[3] < h:
                crop_y0 = randint(0, bounding_box[1])
                crop_h = randint(bounding_box[3], h) - crop_y0

                img = crop(
                    img,
                    crop_y0,
                    0,
                    crop_h,
                    w,
                )

                y0 = bounding_box[1] - crop_y0
                y1 = bounding_box[3] - crop_y0

        return img, asarray([x0, y0, x1, y1])


class LocReshape:
    """Random reshape.

    Represents random reshape, which transforms image and bounding box.
    """

    def __init__(
        self, p: float = 0.5, reshape_lower: float = 0.5, reshape_upper: float = 2
    ):
        """Initializes a random crop.

        Parameters
        ----------
        p : float
            Probability with which transformation is applied.
        reshape_lower : float
            Lower bound for ratio by which is the size of an image multiplied.
        reshape_upper : float
            Upper bound for ratio by which is the size of an image multiplied.
        """
        self.p = p
        self.reshape_lower = reshape_lower
        self.reshape_upper = reshape_upper

    def __call__(self, img: Image, bounding_box: array) -> tuple:
        """Applies random crop on an image and a bounding box.

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
        if uniform() <= self.p:
            w, h = img.size
            ratio = uniform(self.reshape_lower, self.reshape_upper)

            w_reshaped, h_reshaped = int(w * ratio), int(h * ratio)
            img = Resize((h_reshaped, w_reshaped))(img)
            bounding_box = asarray(bounding_box) * ratio

        return img, bounding_box


class LocCompose:
    def __init__(self, transformations: list):
        """Initializes a composition of custom transformations.

        Parameters
        ----------
        transformations: list
            List of transformations to apply.
        """
        self.transformations = transformations

    def __call__(self, img: Image, bounding_box: tuple) -> tuple:
        """Applies transformations on an image and a bounding box.

        Parameters
        ----------
        img : Image
            Image to be processed.
        bounding_box : tuple
            Bounding box to be processed.
        
        Returns
        -------
        tuple
            Processed image and bounding box.
        """
        for transformation in self.transformations:
            img, bounding_box = transformation(img, bounding_box)

        return img, bounding_box
