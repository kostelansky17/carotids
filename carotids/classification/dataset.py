from os import listdir

from torch import tensor
from torch.utils.data.dataset import Dataset

from carotids.preprocessing import load_img


class ClassificationDataset(Dataset):
    """Represents a dateset used for classification.

    Reads names of the files for all classes and creates labels.
    Loads the image when it is requested.
    """

    def __init__(self, img_dirs: dict, transformations: list) -> None:
        """Initializes a classification dataset. The img_dirs
        should contain class as a key and path to the folder
        containing the samples as a value.

        Parameters
        ----------
        img_dirs : dict
            Dictionary with labels as keys and paths to data as values.
        transformations : list
            List of transformations used to preprocess the image inputs.
        """
        self.data_files, self.labels = self._prepare_data(img_dirs)

        self.img_dirs = img_dirs
        self.transformations = transformations

    def _prepare_data(self, img_dirs: dict) -> tuple:
        """Prepares the names of the inputs and the labels.

        Parameters
        ----------
        img_dirs : dict
            Dictionary with labels as keys and paths to data as values.

        Returns
        -------
        tuple
            List of data files names and list of labels.
        """
        data_files = []
        labels = []

        for key in img_dirs:
            img_names = sorted(listdir(img_dirs[key]))
            data_files.extend(img_names)
            labels.extend([key] * len(img_names))

        return data_files, labels

    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at the specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.

        Returns
        -------
        tuple
            Transformed and preprocessed image into a tensor with a label.
        """
        label = self.labels[index]

        img = load_img(self.img_dirs[label], self.data_files[index])
        return self.transformations(img), tensor(label)

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.labels)
