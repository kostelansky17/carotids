from json import load
from os import listdir

from numpy import asarray
from torch import as_tensor, int64, tensor
from torch.utils.data.dataset import Dataset

from carotids.localization.transformations import transform_item
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
    with open(COCO) as coco_file:
        coco_data = load(coco_file)

    labels = {}
    number_of_labels = lencoco_data

    for i in range(number_of_labels):
        file_name = coco_data["images"][i]["file_name"]
        data = coco_data["annotations"][i]

        boxes = [
            asarray(
                [
                    data["bbox"][0],
                    data["bbox"][1],
                    data["bbox"][0] + data["bbox"][2],
                    data["bbox"][1] + data["bbox"][3],
                ]
            )
        ]

        labels[file_name] = {
            "boxes": boxes,
            "labels": tensor([1], dtype=int64),
            "image_id": tensor([data["id"]], dtype=int64),
            "area": data["area"],
        }

    return labels


class FastCarotidDatasetBrno(Dataset):
    """Represents a dateset used for training Faster R-CNN.

    Reads names od the images and labels from Brno dataset. The data (image and
    label file) are loaded when an item is being gotten.
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


class FastCarotidDatasetPrague(Dataset):
    """Represents a dateset used for training Faster R-CNN.

    Reads names od the images and labels from Prague dataset. The images are 
    loaded when an item is being gotten and the labels are red when the Dataset
    is initialized.
    """

    def __init__(self, data_path: str, labels_path: str, transformations: list) -> None:
        """Initializes a training Faster R-CNN dataset.

        Parameters
        ----------
        data_path : str
            Path to a folder containing the images.
        labels_path:str
            Path to a JSON file containing the COCO labels.
        transformations : list
            List of torch transformations used to preprocess the image inputs.
        """
        self.data_path = data_path
        self.labels_path = labels_path

        self.data_files = sorted(listdir(data_path))
        self.labels_path = listdir(labels_path)
        self.labels = load_coco_labels(self.labels_path)

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
            Image processed into a tensor and a dictionary decribing objects'
            bounding boxes, labels, id of the image and objects' areas.
        """
        img = load_img(self.data_path, self.data_files[index])
        label = self.labels(self.data_files[index])

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
