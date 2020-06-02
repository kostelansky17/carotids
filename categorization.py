import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import transforms
from torchvision import models, transforms

from carotids.categorization.categorization_cnn import create_small_cnn, create_vgg
from carotids.categorization.categorization_dataset import CategorizationDataset
from carotids.train_model import train_model


TRAIN_IMG_DIRS = {
    0: "/home/martin/Documents/cartroids/data/categorization/train/diff",
    1: "/home/martin/Documents/cartroids/data/categorization/train/long",
    2: "/home/martin/Documents/cartroids/data/categorization/train/trav",
}
TEST_IMG_DIRS = {
    0: "/home/martin/Documents/cartroids/data/categorization/test/diff",
    1: "/home/martin/Documents/cartroids/data/categorization/test/long",
    2: "/home/martin/Documents/cartroids/data/categorization/test/trav",
}
CATEGORIES = 3
SMALL_TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.1147, 0.1146, 0.1136], [0.0183, 0.0181, 0.0182]),
    ]
)
BIG_TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1145, 0.1144, 0.1134], [0.1694, 0.1675, 0.1684]),
    ]
)


def logit_categorization(train_dataset, test_dataset):
    pass


def cnn_categorization(train_dataset, test_dataset, model):
    model = create_small_cnn(CATEGORIES)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model, losses, accuracies = train_model(
        model, train_dataset, loss, optimizer, device, scheduler
    )


def main():
    train_dataset = CategorizationDataset(TRAIN_IMG_DIRS, SMALL_TRANSFORMATIONS)
    test_dataset = CategorizationDataset(TEST_IMG_DIRS, SMALL_TRANSFORMATIONS)

    model = create_small_cnn(CATEGORIES)
    model = create_vgg(CATEGORIES)

    model, losses, accuracies = (
        cnn_categorization(train_dataset, test_dataset, model),
    )


if __name__ == "__main__":
    main()
