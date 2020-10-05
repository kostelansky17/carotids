import sys

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import models, transforms

from carotids.categorization.models import create_vgg
from carotids.categorization.dataset import CategorizationDataset
from carotids.metrics import accuracy_dataset
from carotids.train_model import train_model
from carotids.utils import GaussianNoiseTransform


TRAIN_IMG_DIRS = {
    0: "FILL_ME",
    1: "FILL_ME",
    2: "FILL_ME",
}
VAL_IMG_DIRS = {
    0: "FILL_ME",
    1: "FILL_ME",
    2: "FILL_ME",
}
TEST_IMG_DIRS = {
    0: "FILL_ME",
    1: "FILL_ME",
    2: "FILL_ME",
}
CATEGORIES = 3

TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1141, 0.1174, 0.1208], [0.1315, 0.1349, 0.1401]),
    ]
)


def main():
    train_dataset = CategorizationDataset(TRAIN_IMG_DIRS, TRANSFORMATIONS)
    val_dataset =  CategorizationDataset(VAL_IMG_DIRS, TRANSFORMATIONS)
    test_dataset = CategorizationDataset(TEST_IMG_DIRS, TRANSFORMATIONS)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = create_vgg(CATEGORIES)
    model.to(device)

    loss = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.99)
    scheduler = MultiStepLR(optimizer, milestones=[15, 30, 60, 90], gamma=0.1)

    model, losses, accuracies = train_model(
        model, train_loader, val_loader, loss, optimizer, device, scheduler, 100
    )
    
    torch.save(model.state_dict(), "categorization_model.pth")


if __name__ == "__main__":
    main()
