from os import listdir

from torch import as_tensor, int64, tensor
from torch.utils.data.dataset import Dataset

from carotids.localization.transformations import transform_item
from carotids.preprocessing import load_img, load_position


class FastCarotidDataset(Dataset):
    """Represents a dateset used for training Faster R-CNN.

    Reads names od the images and labels. The data are loaded when an item is
    gotten.
    """

    def __init__(
        self,
        data_path: str,
        labels_path: str,
        transformations_custom: list,
        transformations_torch: list,
    ) -> None:
        """Initializes a training Faster R-CNN dataset.

        Parameters
        ----------
        data_path : str
            Path to a folder containing the images.
        labels_path:str
            Path to a folder containing the files with labels.
        transformations_custom : list
            List of custom transformations used to preprocess the image inputs.
        transformations_torch : list
            List of torch transformations used to preprocess the image inputs.
        """
        self.data_path = data_path
        self.labels_path = labels_path

        self.data_files = sorted(listdir(data_path))
        self.labels_files = sorted(listdir(labels_path))

        self.transformations_custom = transformations_custom
        self.transformations_torch = transformations_torch

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.
        
        Returns
        -------
        tuple
            Image processed into a tensor and a dictionary decribing objects'
            bounding boxes, labels, id of the image and objects' areas.
        """
        img = load_img(self.data_path, self.data_files[index])
        label = load_position(self.labels_path, self.labels_files[index])

        img, label = transform_item(img, label, self.transformations_custom)

        boxes = [label]
        boxes = as_tensor([label], dtype=int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        label = {
            "boxes": boxes,
            "labels": tensor([1], dtype=int64),
            "image_id": tensor([index], dtype=int64),
            "area": area,
        }

        return self.transformations_torch(img), label

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_files)


class FastCarotidDatasetEval(Dataset):
    """Represents a dateset used to store data which are the input into 
    Faster R-CNN.

    Reads names od the images. The data are loaded when an item is
    gotten.
    """

    def __init__(self, data_path: int, transformations: list) -> None:
        """Initializes a training Faster R-CNN dataset.

        Parameters
        ----------
        data_path : str
            Path to a folder containing the images.
        transformations : list
            List of transformations used to preprocess the image inputs.
        """
        self.data_path = data_path
        self.data_files = sorted(listdir(data_path))
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
            Image processed into a tensor and name of the image file.
        """
        img = load_img(self.data_path, self.data_files[index])

        return self.transformations(img), self.data_files[index]

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_files)
