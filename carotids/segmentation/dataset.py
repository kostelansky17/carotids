from os import listdir

from torch import cat, int64, tensor, Tensor, unsqueeze, zeros
from torch.nn.functional import one_hot
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose

from carotids.preprocessing import load_img
from carotids.segmentation.transformations import SegCompose


class SegmentationDataset(Dataset):
    """Represents a dateset used for segmentation.

    Creates a dataset used in the segmentation. Assumes
    that original images and segmentation masks are in
    the separate folders.
    """

    def __init__(
        self,
        data_path: str,
        labels_path: str,
        transformations_custom: SegCompose,
        transformations_torch: Compose,
        plaque_with_wall: bool = False,
        encode_to_one_hot: bool = True,
    ) -> None:
        """Initializes a segmentation dataset.

        Parameters
        ----------
        data_path : str
            Folder with raw images.
        labels_path : str
            Folder with references
        transformations_custom : SegCompose
            Composition of custom segmentation transformations.
        transformations_torch : Compose
            Composition of torch segmentation transformations.
        plaque_with_wall : bool
            If True, the plaque and wall classes are united.
        encode_to_one_hot : bool
            If true, the reference is returned encoded in a one-hot setting,
            if false, the reference is returned with the classes encoded as an
            integer values.
        """
        self.data_path = data_path
        self.labels_path = labels_path
        self.img_files = sorted(listdir(data_path))

        self.transformations_custom = transformations_custom
        self.transformations_torch = transformations_torch

        self.plaque_with_wall = plaque_with_wall
        self.encode_to_one_hot = encode_to_one_hot

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.

        Returns
        -------
        tuple
            Returns the processed image and the transformed segmentation mask
        """
        img = load_img(self.data_path, self.img_files[index])
        label = load_img(self.labels_path, self.img_files[index])

        img, label = self.transformations_custom(img, label)
        img, label = self.transformations_torch(img), self.transformations_torch(label)

        label = self._label_to_mask(label)

        return img, label

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.img_files)

    def _label_to_mask(self, label: Tensor):
        """Processes label to a Tensor mask. If plaque_with_wall
        is selected to True, the plaque category is changed
        to the wall category. If encode_to_one_hot is selected to True,
        the mask is encoded in a one-hot setting.

        Parameters
        ----------
        label : Tensor
            Index of an item to return.

        Returns
        -------
        Tensor
            Tensor label.
        """
        mask = cat((zeros(1, 512, 512), label))
        mask = mask.argmax(0)

        if self.plaque_with_wall:
            mask[mask == 3] = 1

        if self.encode_to_one_hot:
            mask = one_hot(mask).permute(2, 0, 1)

        return mask


class SegmentationEvaluationDataset(Dataset):
    """Represents a dateset used to create segmentation masks of the images.

    Creates a dataset used to create segmentation of the images. Assumes
    that the images in the specified folder contain only the region
    with an artery.
    """

    def __init__(
        self, data_path: str, transformations: Compose, labels_path: str = None
    ) -> None:
        """Initializes an evaluation segmentation dataset.

        Parameters
        ----------
        data_path : str
            Folder with the raw images.
        transformations : Compose
            Composition of torch image transformations.
        labels_path : str
            Folder with the references.
        """
        self.data_path = data_path
        self.labels_path = labels_path

        self.img_files = sorted(listdir(data_path))
        self.transformations = transformations

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.

        Returns
        -------
        tuple
            Returns the processed image, the the original one, and if the folder
            with the references was defined, the ground truth is returned as 
            well as an image.
        """
        img = load_img(self.data_path, self.img_files[index])
        img_torch = self.transformations(img)

        if self.labels_path is not None:
            label = load_img(self.data_path, self.img_files[index])

            return img_torch, img, label

        return img_torch, img

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.img_files)
