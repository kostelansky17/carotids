import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import transforms
from torchvision import models, transforms

from carotids.categorization.categorization_cnn import create_small_cnn, create_vgg
from carotids.categorization.categorization_dataset import CategorizationDataset
from carotids.categorization.categorization_linear import (
    create_categorization_features,
    train_classifier,
    try_hyperparameters,
)
from carotids.metrics import accuracy_np, accuracy_torch
from carotids.train_model import train_model
from carotids.metrics import accuracy_np


TRAIN_IMG_DIRS = {
    0: "/home/martin/Documents/cartroids/data/categorization/train/long",
    1: "/home/martin/Documents/cartroids/data/categorization/train/trav",
    2: "/home/martin/Documents/cartroids/data/categorization/train/diff",
}
TEST_IMG_DIRS = {
    0: "/home/martin/Documents/cartroids/data/categorization/test/long",
    1: "/home/martin/Documents/cartroids/data/categorization/test/trav",
    2: "/home/martin/Documents/cartroids/data/categorization/test/diff",
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


def logreg_categorization(train_img_dirs, test_img_dirs):
    X_train, y_train = create_categorization_features(train_img_dirs)
    X_test, y_test = create_categorization_features(test_img_dirs)

    Cs = [{"C": c} for c in np.logspace(0.01, 1, 10, endpoint=True)]
    test_accs, best_C = try_hyperparameters(X_train, y_train, Cs)

    clf = train_classifier(X_train, y_train, best_C)
    test_acc = accuracy_np(clf.predict(X_test), y_test)

    print(f"Test accuracy: {test_acc}")

    return clf


def cnn_categorization(model):
    train_dataset = CategorizationDataset(TRAIN_IMG_DIRS, SMALL_TRANSFORMATIONS)
    test_dataset = CategorizationDataset(TEST_IMG_DIRS, SMALL_TRANSFORMATIONS)

    model = create_small_cnn(CATEGORIES)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model, losses, accuracies = train_model(
        model, train_dataset, loss, optimizer, device, scheduler
    )

    return model, losses, accuracies


def main():
    # model = create_small_cnn(CATEGORIES)
    # model = create_vgg(CATEGORIES)
    # model, losses, accuracies = cnn_categorization(model)

    logreg_categorization(TRAIN_IMG_DIRS, TEST_IMG_DIRS)


if __name__ == "__main__":
    main()
