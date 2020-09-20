from os import listdir

from torch.utils.data.dataset import Dataset

from carotids.preprocessing import load_img, load_position
from carotids.utils import recompute_labels


class ResnetCarotidDataset(Dataset):
    """Represents a dateset used for training localization CNN.

    Reads names od the images and labels. The data are loaded when an item is
    gotten.
    """

    def __init__(self, data_path: str, labels_path: str, transformations: list) -> None:
        """Initializes a training CNN used for localization.

        Parameters
        ----------
        data_path : str
            Path to a folder containing the images.
        labels_path:str
            Path to a folder containing the files with labels.
        transformations : list
            List of transformations used to preprocess the image inputs.
        """
        self.data_path = data_path
        self.labels_path = labels_path

        self.data_files = sorted(listdir(data_path))
        self.labels_files = sorted(listdir(labels_path))

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
            Image processed into a tensor and a tensor with coordinates of a 
            bounding box.
        """
        img = load_img(self.data_path, self.data_files[index])
        x0, y0, x1, y1 = load_position(self.labels_path, self.labels_files[index])
        label_tensor = recompute_labels(img, x0, y0, x1, y1)

        return self.transformations(img).double(), label_tensor.double()

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_files)
