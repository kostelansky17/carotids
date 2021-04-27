from os import listdir

from torch import cat, int64, Tensor, unsqueeze, zeros
from torch import DataLoader
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose

from carotids.preprocessing import load_img
from carotids.segmentation.transformations import SegCompose
from carotids.utils import split_dataset, split_dataset_into_dataloaders


def label_to_mask(
    label: Tensor, plaque_with_wall: bool = False, encode_to_one_hot: bool = True
):
    """Processes label to a Tensor mask. If plaque_with_wall
    is selected to True, the plaque category is changed
    to the wall category. If encode_to_one_hot is selected to True,
    the mask is encoded in a one-hot setting.

    Parameters
    ----------
    label : Tensor
        Index of an item to return.
    plaque_with_wall : bool
        If True, the plaque and wall classes are united.
    encode_to_one_hot : bool
        If true, the reference is returned encoded in a one-hot setting,
        if false, the reference is returned with the classes encoded as an
        integer values.

    Returns
    -------
    Tensor
        Tensor label.
    """
    mask = cat((zeros(1, *label.shape[1:]), label))
    mask = mask.argmax(0)

    if plaque_with_wall:
        mask[mask == 3] = 1

    if encode_to_one_hot:
        mask = one_hot(mask).permute(2, 0, 1)
        mask[0, (label[0, ...] == 1) & (label[1, ...] == 1)] = 1

    return mask


class SegmentationDatamodule:
    """Represent Datamodule used in the segmentation. This class shatters all of
    the datasets used for training and evaluation - train, validation, and test.
    """

    def __init__(
        self,
        imgs_path: str,
        labels_path: str,
        trans_seg_train: SegCompose,
        trans_seg_val: SegCompose,
        trans_torch: Compose,
        batch_size: int = 2,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> None:
        """Initializes a segmentation datamodule.

        Parameters
        ----------
        imgs_path : str
            Folder with raw images.
        labels_path : str
            Folder with references
        trans_seg_train : SegCompose
            Composition of custom segmentation transformations for training.
        trans_seg_val : SegCompose
            Composition of custom segmentation transformations for evaluation.
        trans_torch : Compose
            Composition of torch segmentation transformations.
        batch_size : int
            Number of samples in the training samples.
        val_split : float
            Precentage of the training data split as the validation set.
        test_split : float
            Precentage of the data split as the test set.
        """
        dataset = SegmentationDataset(
            imgs_path,
            labels_path,
            trans_seg_train,
            trans_torch,
            False,
            True,
        )

        dataset_simple = SegmentationDataset(
            imgs_path,
            labels_path,
            trans_seg_val,
            trans_torch,
            False,
            True,
        )

        train_set, _, _, _ = split_dataset(dataset, test_split)
        train_set_simple, _, test_set, _ = split_dataset(dataset_simple, test_split)

        self.train_loader, _, _, _ = split_dataset_into_dataloaders(
            train_set, val_split, batch_size
        )
        self.train_loader_simple, _, self.val_loader, _ = split_dataset_into_dataloaders(
            train_set_simple, val_split, batch_size
        )
        self.test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        dataset_eval = SegmentationEvaluationDataset(
            imgs_path,
            labels_path,
            trans_seg_val,
            trans_torch,
            False,
            True,
        )

        train_val_set_eval, _, self.test_set_eval, _ = split_dataset(
            dataset_simple, test_split
        )
        self.train_set_eval, _, self.val_set_eval, _ = split_dataset(
            train_val_set_eval, val_split
        )

    def get_train_loader(self) -> DataLoader:
        """Returns train DataLoader."""
        self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """Returns validation DataLoader."""
        self.val_loader

    def get_test_loader(self) -> DataLoader:
        """Returns test DataLoader."""
        self.test_loader

    def get_train_eval_loader(self) -> DataLoader:
        """Returns training dataset without the data augmentation."""
        return self.train_loader_simple

    def get_evaluation_datasets(self) -> tuple:
        """Returns triple of datasets used for evaluation and visualization. The
        order of the datasets is - Training, Validation, Test.
        """
        return self.train_set_eval, self.val_set_eval, self.test_set_eval


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

        label = label_to_mask(label, self.plaque_with_wall, self.encode_to_one_hot)

        return img, label

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.img_files)


class SegmentationEvaluationDataset(Dataset):
    """Represents a dateset used to create segmentation masks of the images.

    Creates a dataset used to create segmentation of the images. Assumes
    that the images in the specified folder contain only the region
    with an artery.
    """

    def __init__(
        self,
        data_path: str,
        transformations_torch: Compose,
        transformations_custom: SegCompose = None,
        labels_path: str = None,
        plaque_with_wall: bool = False,
        encode_to_one_hot: bool = True,
    ) -> None:
        """Initializes an evaluation segmentation dataset.

        Parameters
        ----------
        data_path : str
            Folder with the raw images.
        transformations_torch : Compose
            Composition of torch segmentation transformations.
        transformations_custom : SegCompose
            Composition of custom segmentation transformations.
        labels_path : str
            Folder with the references.
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

        self.transformations_torch = transformations_torch
        self.transformations_custom = transformations_custom

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
            Returns the processed image, the original one, its name, and if the
            folder with the references was defined, the ground truth is returned
            as well as an image.
        """
        img = load_img(self.data_path, self.img_files[index])

        if self.labels_path is not None:
            label = load_img(self.labels_path, self.img_files[index])

            if self.transformations_custom is not None:
                img, label = self.transformations_custom(img, label)

            img_torch = self.transformations_torch(img)
            label_torch = self.transformations_torch(label)

            label_torch = label_to_mask(
                label_torch, self.plaque_with_wall, self.encode_to_one_hot
            )

            return img_torch, img, self.img_files[index], label_torch, label

        else:
            img_torch = self.transformations_torch(img)

            return img_torch, img, self.img_files[index]

    def __len__(self) -> int:
        """Returns a length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.img_files)
