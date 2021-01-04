from json import load
from os import listdir

from numpy import asarray
from torch import as_tensor, int64, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose

from carotids.localization.transformations import LocCompose
from carotids.preprocessing import load_img, load_position


def load_coco_labels(labels_path: str) -> dict:
    """Reads JSON file with COCO labels and transforms them into a dictionary.

    Parameters
    ----------
    labels_path : str
        A path to JSON file with COCO labels.

    Returns
    -------
    dict
        Dictionary with labels, key is the image file name and values are
        dictionaries with data needed in training.
    """
    with open(labels_path) as coco_file:
        coco_data = load(coco_file)

    labels = {}
    images = coco_data["images"]
    annotations = coco_data["annotations"]

    for annotation in annotations:
        if annotation["category_id"] == 1:
            file_name = images[annotation["image_id"] - 1]["file_name"]
            box = [
                annotation["bbox"][0],
                annotation["bbox"][1],
                annotation["bbox"][0] + annotation["bbox"][2],
                annotation["bbox"][1] + annotation["bbox"][3],
            ]

            labels[file_name] = {
                "box": box,
                "labels": [1],
                "image_id": annotation["image_id"],
            }

    return labels


class FastCarotidDatasetSPLab(Dataset):
    """Represents a dateset used for training Faster R-CNN.

    Reads names od the images and labels from SPLab dataset. The data (image and
    label file) are loaded when an item is being gotten. The lables and images
    are in the separate folders. Both folder should contain the same number
    of files. Name of a label and an image needs to be the same for a sample.
    """

    def __init__(
        self,
        data_path: str,
        labels_path: str,
        transformations_custom: LocCompose,
        transformations_torch: Compose,
    ) -> None:
        """Initializes a training Faster R-CNN dataset.

        Parameters
        ----------
        data_path : str
            Path to a folder containing the images.
        labels_path:str
            Path to a folder containing the files with labels.
        transformations_custom : Compose
            List of custom transformations used to preprocess the image inputs.
        transformations_torch : LocCompose
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

        img, label = self.transformations_custom(img, label)

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


class FastCarotidDatasetANTIQUE(Dataset):
    """Represents a dateset used for training Faster R-CNN.

    Reads names od the images and labels from ANTIQUE dataset. The images are
    loaded when an item is being gotten and the labels are red when the Dataset
    is initialized. The images are expected to be in a separate folder and
    the path to JSON files with labels in COCO format need to be provided.
    """

    def __init__(
        self,
        data_path: str,
        labels_path: str,
        transformations_custom: LocCompose,
        transformations_torch: Compose,
    ) -> None:
        """Initializes a training Faster R-CNN dataset.

        Parameters
        ----------
        data_path : str
            Path to a folder containing the images.
        labels_path:str
            Path to a JSON file containing the COCO labels.
        transformations_custom : Compose
            List of custom transformations used to preprocess the image inputs.
        transformations_torch : LocCompose
            List of torch transformations used to preprocess the image inputs.
        """
        self.data_path = data_path
        self.labels_path = labels_path

        self.data_files = sorted(listdir(data_path))
        self.labels_path = labels_path
        self.labels = load_coco_labels(self.labels_path)

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
        label = self.labels[self.data_files[index]]

        img, box = self.transformations_custom(img, label["box"])

        boxes = as_tensor([box], dtype=int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        label_out = {
            "boxes": boxes,
            "labels": tensor([1], dtype=int64),
            "image_id": tensor([index], dtype=int64),
            "area": area,
        }

        return self.transformations_torch(img), label_out

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
    Faster R-CNN. Reads all files in a folder. The data are loaded when an
    item is gotten.
    """

    def __init__(self, data_path: int, transformations: Compose) -> None:
        """Initializes a training Faster R-CNN dataset.

        Parameters
        ----------
        data_path : str
            Path to a folder containing the images.
        transformations : Compose
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
