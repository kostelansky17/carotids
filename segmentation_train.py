import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
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
from carotids.segmentation.transformations import (
    SegCompose,
    SegCrop,
    SegRandomHorizontalFlip,
    SegRandomVerticalFlip,
)
from carotids.train_model import train_model


IMG_PATH = ""
LABELS_PATH = ""
CATEGORIES = 4

TRANSFORMATIONS_TRAIN_SEG = SegCompose(
    [
        SegRandomHorizontalFlip(),
        SegRandomVerticalFlip(),
        SegCrop(15),
   ]
)
TRANSFORMATIONS_TEST_SEG = SegCompose(
    [
        SegCrop(),
   ]
)
TRANSFORMATIONS_TORCH = Compose(
    [
        Resize((512, 512)),
        ToTensor(),
    ]
)


def main():
    """The approach which has been used for training best categorization model
    as defined and described in the given report.
    """
    train_dataset = SegmentationDataset(
        IMG_PATH, 
        LABELS_PATH,
        TRANSFORMATIONS_TRAIN_SEG,
        TRANSFORMATIONS_TORCH
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Unet(CATEGORIES)
    model.to(device)

    loss = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.99)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)

    model, losses, accuracies = train_model(
        model, train_loader, train_loader, loss, optimizer, device, scheduler, 1
    )

    torch.save(model.state_dict(), "categorization_model.pth")


if __name__ == "__main__":
    main()
