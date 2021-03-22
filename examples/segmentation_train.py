from torch import cuda, device, save
from torch.nn import CrossEntropyLoss, Module
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

from carotids.segmentation.dataset import SegmentationDataset
from carotids.segmentation.model import Unet
from carotids.segmentation.train_model import train_model
from carotids.segmentation.transformations import (
    SegCompose,
    SegCrop,
    SegRandomHorizontalFlip,
    SegRandomVerticalFlip,
)


CATEGORIES = 3
EPOCHS = 1

# data pats - longitudinal
LONGITUDINAL_TRAIN_IMG_PATH = "INSERT_PATH"
LONGITUDINAL_TRAIN_LABELS_PATH = "INSERT_PATH"

LONGITUDINAL_VAL_IMG_PATH = "INSERT_PATH"
LONGITUDINAL_VAL_LABELS_PATH = "INSERT_PATH"

# data pats - transverse
TRANSVERSE_TRAIN_IMG_PATH = "INSERT_PATH"
TRANSVERSE_TRAIN_LABELS_PATH = "INSERT_PATH"

TRANSVERSE_VAL_IMG_PATH = "INSERT_PATH"
TRANSVERSE_VAL_LABELS_PATH = "INSERT_PATH"


# transformations
TRANSFORMATIONS_SEG = SegCompose(
    [
        SegCrop(default_t=15),
    ]
)
TRANSFORMATIONS_TORCH = Compose(
    [
        Resize((512, 512)),
        ToTensor(),
    ]
)


def train_segmentation_model(
    TRAIN_IMG_PATH: str,
    TRAIN_LABELS_PATH: str,
    VAL_IMG_PATH: str,
    VAL_LABELS_PATH: str,
    model_save_name: str = "segmentation_model.pt",
) -> Module:
    """The approach which has been used for training best categorization model
    as defined and described in the given report.

    Parameters
    ----------
    TRAIN_IMG_PATH : str
        Path to raw train images.
    TRAIN_LABELS_PATH : str
        Path to train labels.
    VAL_IMG_PATH : str
        Path to raw validation images.
    VAL_LABELS_PATH : str
        Path to train labels.
    model_save_name : str
        Name of the trained model.

    Returns
    -------
    Module
        Trained model.
    """
    train_dataset = SegmentationDataset(
        TRAIN_IMG_PATH, TRAIN_LABELS_PATH, TRANSFORMATIONS_SEG, TRANSFORMATIONS_TORCH
    )

    val_dataset = SegmentationDataset(
        VAL_IMG_PATH, VAL_LABELS_PATH, TRANSFORMATIONS_SEG, TRANSFORMATIONS_TORCH
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    torch_device = device("cuda") if cuda.is_available() else device("cpu")

    model = Unet(CATEGORIES)
    model.to(torch_device)

    loss = CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=0.0001, momentum=0.99)
    scheduler = ReduceLROnPlateau(optimizer, patience=5)

    model = train_model(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        torch_device,
        scheduler,
        EPOCHS,
    )

    save(model.state_dict(), model_save_name)


if __name__ == "__main__":
    print("Transverse model:")
    train_segmentation_model(
        TRANSVERSE_TRAIN_IMG_PATH,
        TRANSVERSE_TRAIN_LABELS_PATH,
        TRANSVERSE_VAL_IMG_PATH,
        TRANSVERSE_VAL_LABELS_PATH,
        "trav_segmentation_model.pt",
    )
    print("Longitudinal model:")
    train_segmentation_model(
        LONGITUDINAL_TRAIN_IMG_PATH,
        LONGITUDINAL_TRAIN_LABELS_PATH,
        LONGITUDINAL_VAL_IMG_PATH,
        LONGITUDINAL_VAL_LABELS_PATH,
        "long_segmentation_model.pt",
    )
