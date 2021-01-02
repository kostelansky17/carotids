from numpy import asarray
from PIL import Image
from torch import FloatTensor, manual_seed, randn, sqrt, tensor
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset


class GaussianNoiseTransform(object):
    """Adds random gaussian noise to a tensor.

    This class can be used in the PyTorch's  tensor.Compose pipeline used for
    preprocessing tensors.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """Initializes a Gaussian noise transformation.

        Parameters
        ----------
        mean : float
            Mean of the distribution used noise generation.
        std : float
            Standart deviation of the distribution used noise generation.
        """
        self.std = std
        self.mean = mean

    def __call__(self, tensor: tensor) -> tensor:
        """Method which allows to call an instance of the class like function.
        Applies transformation on a tensor.

        Parameters
        ----------
        tensor : tensor
            Tensor to apply transformation on.

        Returns
        -------
        tensor
            Transformed tensor.
        """
        return tensor + randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        """Creates string representation of the GaussianNoiseTransform object.

        Returns
        -------
        str
            String representation of the GaussianNoiseTransform object.
        """
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def compute_mean_image_dataloader(dataloader: DataLoader) -> tensor:
    """Computes mean of an image dataloader (per channel).

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader to compute mean.

    Returns
    -------
    tensor
        Mean of the dataloader.
    """
    mean = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)

    return mean


def compute_std_image_dataloader(dataloader: DataLoader, mean: tensor) -> tensor:
    """Computes std of an image dataloader (per channel).

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader to compute standart deviation.
    mean : tensor
        Mean of the dataloader.

    Returns
    -------
    tensor
        Standart deviation of the dataloader.
    """
    var = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = sqrt(var / (len(dataloader.dataset) * 224 * 224))

    return std


def compute_standardization_image_dataloader(dataloader: DataLoader) -> tuple:
    """Computes mean and standardization parameters of a dataloader.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader to compute parameters.

    Returns
    -------
    tuple
        Mean and standart deviation of the dataloader.
    """
    mean = compute_mean_image_dataloader(dataloader)
    std = compute_std_image_dataloader(dataloader, mean)

    return mean, std


def split_dataset_into_dataloaders(
    dataset: Dataset, val_split: float = 0.1, batch_size: int = 64, seed: int = 17
):
    """Splits dataset into train and validation ones and transform them into
    dataloaders.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    val_split : float
        Ratio used in spliting.
    batch_size : int
        Size of batches in dataloaders.
    seed : int
        Seed used for spliting samples

    Returns
    -------
    tuple
        Train dataloader with number of train samples and validation dataloader
        with number of validation samples.
    """
    trainset, train_size, valset, val_size = split_dataset(dataset, val_split, seed)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return train_loader, train_size, val_loader, val_size


def split_dataset(dataset: Dataset, val_split: float = 0.1, seed: int = 17):
    """Splits dataset into train and validation ones and transform them into
    dataloaders.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    val_split : float
        Ratio used in spliting.
    seed : int
        Seed used for spliting samples

    Returns
    -------
    tuple
        Train dataloader with number of train samples and validation dataloader
        with number of validation samples.
    """
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    manual_seed(seed)
    trainset, valset = random_split(dataset, [train_size, val_size])

    return trainset, train_size, valset, val_size


def recompute_labels(
    image: Image, x0: int, y0: int, x1: int, y1: int, target_size: tuple = (244, 244)
) -> FloatTensor:
    """Splits dataset into train and validation ones and transform them into
    dataloaders.

    Parameters
    ----------
    image : Image
        Dataset to split.
    x0 : int
        X coordinate of left upper corner of the object's bounding box.
    y0 : int
        Y coordinate of left upper corner of the object's bounding box.
    x1 : int
        X coordinate of right lower corner of the object's bounding box.
    y1 : int
        Y coordinate of right lower corner of the object's bounding box.
    target_size : tuple
        The shape into which needs to be coordinates recomputed.

    Returns
    -------
    FloatTensor
        Coordinates of bounding box recomputed to the shape into which will be
        the image reshaped.
    """
    if len(asarray(image).shape) == 2:
        img_height, img_width = asarray(image).shape
    else:
        img_height, img_width, _ = asarray(image).shape

    target_height, target_width = target_size

    x0, x1 = x0 * (img_width / target_width), x1 * (img_width / target_width)
    y0, y1 = y0 * (img_height / target_height), y1 * (img_height / target_height)

    return FloatTensor([x0, y0, x1, y1])
