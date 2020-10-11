import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms

from carotids.categorization.models import create_vgg
from carotids.categorization.dataset import CategorizationDataset
from carotids.train_model import train_model


TRAIN_IMG_DIRS = {0: "FILL_ME", 1: "FILL_ME", 2: "FILL_ME"}
VAL_IMG_DIRS = {0: "FILL_ME", 1: "FILL_ME", 2: "FILL_ME"}
TEST_IMG_DIRS = {0: "FILL_ME", 1: "FILL_ME", 2: "FILL_ME"}
CATEGORIES = 3

TRANSFORMATIONS_TRAIN = transforms.Compose(
    [
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.1257, 0.1267, 0.1278], [0.1528, 0.1537, 0.1551]),
        GaussianNoiseTransform(std=0.001),
    ]
)
TRANSFORMATIONS_TEST = transforms.Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.1257, 0.1267, 0.1278], [0.1528, 0.1537, 0.1551]),
    ]
)


def main():
    """The approach which has been used for training best categorization model
    as defined and described in the given report.
    """
    train_dataset = CategorizationDataset(TRAIN_IMG_DIRS, TRANSFORMATIONS_TRAIN)
    val_dataset = CategorizationDataset(VAL_IMG_DIRS, TRANSFORMATIONS_TEST)
    test_dataset = CategorizationDataset(TEST_IMG_DIRS, TRANSFORMATIONS_TEST)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = create_vgg(CATEGORIES)
    model.to(device)

    loss = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.99)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60], gamma=0.1)

    model, losses, accuracies = train_model(
        model, train_loader, val_loader, loss, optimizer, device, scheduler, 100
    )

    torch.save(model.state_dict(), "categorization_model.pth")


if __name__ == "__main__":
    main()
