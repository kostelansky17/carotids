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
from carotids.metrics import accuracy_np, accuracy_dataset
from carotids.preprocessing import normalize_data
from carotids.train_model import train_model


TRAIN_IMG_DIRS = {
    0: "/contentdrive/My Drive/cartroids/categorization/train/long",
    1: "/contentdrive/My Drive/cartroids/categorization/train/trav",
    2: "/contentdrive/My Drive/cartroids/categorization/train/diff",
}
TEST_IMG_DIRS = {
    0: "/contentdrive/My Drive/cartroids/categorization/test/long",
    1: "/contentdrive/My Drive/cartroids/categorization/test/trav",
    2: "/contentdrive/My Drive/cartroids/categorization/test/diff",
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
    print("Logistic regression for categorization...")
    X_train, y_train = create_categorization_features(train_img_dirs)
    X_test, y_test = create_categorization_features(test_img_dirs)
    X_train, X_test = normalize_data(X_train, X_test)

    Cs = [{"C": c} for c in np.logspace(0.01, 1, 10, endpoint=True)]
    test_accs, best_C = try_hyperparameters(X_train, y_train, Cs)

    clf = train_classifier(X_train, y_train, best_C)
    test_acc = accuracy_np(clf.predict(X_test), y_test)

    print(f"Test accuracy: {test_acc}")

    return clf


def cnn_categorization(model, transformations):
    train_dataset = CategorizationDataset(TRAIN_IMG_DIRS, transformations)
    test_dataset = CategorizationDataset(TEST_IMG_DIRS, transformations)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model, losses, accuracies = train_model(
        model, train_dataset, loss, optimizer, device, scheduler
    )

    test_accuracy = accuracy_dataset(test_dataset, model, device)

    return model, losses, accuracies, test_accuracy


def small_cnn_categorization():
    print("Small CNN for categorization...")
    
    model = create_small_cnn(CATEGORIES)
    model, losses, accuracies, test_accuracy = cnn_categorization(model, SMALL_TRANSFORMATIONS)
    
    print(f"Test accuracy: {test_accuracy}")
    return model, losses, accuracies, test_accuracy


def big_cnn_categorization():
    print("Big CNN for categorization...")
    
    model = create_vgg(CATEGORIES)
    model, losses, accuracies, test_accuracy = cnn_categorization(model, BIG_TRANSFORMATIONS)
    
    print(f"Test accuracy: {test_accuracy}")
    return model, losses, accuracies, test_accuracy


def main():
    logreg_categorization(TRAIN_IMG_DIRS, TEST_IMG_DIRS)
    small_cnn_categorization()
    big_cnn_categorization()


if __name__ == "__main__":
    main()
