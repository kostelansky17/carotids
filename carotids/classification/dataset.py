from torch.utils.data.dataset import Dataset


class ClassificationDataset(Dataset):
    """Represents a dateset used for classification.
    """

    def __init__(self) -> None:
        """Initializes a classification dataset.

        Parameters
        ----------
        """


    def __getitem__(self, index: int) -> tuple:
        """Gets item from the dataset at a specified index.

        Parameters
        ----------
        index : int
            Index of an item to return.
        
        Returns
        -------
        tuple
            TODO.
        """
        pass

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        pass
