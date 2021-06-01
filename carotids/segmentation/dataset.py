from os import listdir

from torch import cat, int64, Tensor, unsqueeze, zeros
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
        mask[mask == 2] = 1
        mask[mask == 3] = 2

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
        trans_torch_label: Compose = None,
        batch_size: int = 2,
        val_split: float = 0.1,
        test_split: float = 0.1,
        num_workers: int = 8,
        plaque_with_wall: bool = False,
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
        trans_torch_label : Compose
            Composition of torch transformations for preprocessing of the label.
        batch_size : int
            Number of samples in the training samples.
        val_split : float
            Precentage of the training data split as the validation set.
        test_split : float
            Precentage of the data split as the test set.
        num_workers : int
            The number of worker processes used for data loading.
        plaque_with_wall : bool
        """
        dataset = SegmentationDataset(
            imgs_path,
            labels_path,
            trans_seg_train,
            trans_torch,
            trans_torch_label,
            plaque_with_wall,
            True,
        )

        dataset_simple = SegmentationDataset(
            imgs_path,
            labels_path,
            trans_seg_val,
            trans_torch,
            trans_torch_label,
            plaque_with_wall,
            True,
        )

        self.train_val_set, _, _, _ = split_dataset(dataset, test_split)
        self.train_val_set_simple, _, self.test_set, _ = split_dataset(
            dataset_simple, test_split
        )

        self.train_loader, _, _, _ = split_dataset_into_dataloaders(
            self.train_val_set, val_split, batch_size, num_workers=num_workers
        )
        self.train_set_simple, _, self.val_set_simple, _ = split_dataset(self.train_val_set_simple, val_split)

        self.val_loader = DataLoader(
            self.val_set_simple, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.train_eval_loader = DataLoader(
            self.train_set_simple, batch_size=1, shuffle=False, num_workers=num_workers
        )
        self.val_eval_loader = DataLoader(
            self.val_set_simple, batch_size=1, shuffle=False, num_workers=num_workers
        )
        self.test_eval_loader = DataLoader(
            self.test_set, batch_size=1, shuffle=False, num_workers=num_workers
        )

        eval_dataset = SegmentationEvaluationDataset(
            imgs_path,
            trans_torch,
            trans_seg_val,
            labels_path,
            trans_torch_label,
            plaque_with_wall,
            True,
        )

        train_val_eval_set, _, self.test_eval_set, _ = split_dataset(
            eval_dataset, test_split
        )
        self.train_eval_set, _, self.val_eval_set, _ = split_dataset(
            train_val_eval_set, val_split
        )


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
        transformations_torch_label: Compose = None,
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
        transformations_torch_label : Compose
            Composition of torch transformations for preprocessing of the label.
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
        self.transformations_torch_label = transformations_torch_label

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
        if self.transformations_torch_label is None:
            img, label = self.transformations_torch(img), self.transformations_torch(
                label
            )
        else:
            img = self.transformations_torch(img)
            label = self.transformations_torch_label(label)

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
        transformations_torch_label: Compose = None,
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
        transformations_torch_label : Compose
            Composition of torch transformations for preprocessing of the label.
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
        self.transformations_torch_label = transformations_torch_label

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

            if self.transformations_torch_label is None:
                img_torch = self.transformations_torch(img)
                label_torch = self.transformations_torch(label)
            else:
                img_torch = self.transformations_torch(img)
                label_torch = self.transformations_torch_label(label)

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
