import sys

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import transforms
from torchvision import models, transforms

from carotids.categorization.categorization_cnn import create_small_cnn, create_vgg, create_resnet50, create_resnet101
from carotids.categorization.categorization_dataset import CategorizationDataset
from carotids.categorization.categorization_linear import (
    create_categorization_features,
    train_classifier,
    try_hyperparameters,
)
from carotids.metrics import accuracy_np, accuracy_dataset
from carotids.preprocessing import normalize_data
from carotids.train_model import train_model
from carotids.utils import GaussianNoiseTransform


TRAIN_IMG_DIRS = {
    0: "/content/drive/My Drive/cartroids/categorization2/train/long",
    1: "/content/drive/My Drive/cartroids/categorization2/train/trav",
    2: "/content/drive/My Drive/cartroids/categorization2/train/diff",
}
TEST_IMG_DIRS = {
    0: "/content/drive/My Drive/cartroids/categorization2/test/long",
    1: "/content/drive/My Drive/cartroids/categorization2/test/trav",
    2: "/content/drive/My Drive/cartroids/categorization2/test/diff",
}
CATEGORIES = 3
COMPLEX_SMALL_TRANSFORMATIONS_TRAIN = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.1147, 0.1146, 0.1136], [0.0183, 0.0181, 0.0182]),
        GaussianNoiseTransform(std=0.001),
    ]
)
SIMPLE_SMALL_TRANSFORMATIONS_TRAIN = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.1147, 0.1146, 0.1136], [0.0183, 0.0181, 0.0182]),
    ]
)
SMALL_TRANSFORMATIONS_TEST = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0.1147, 0.1146, 0.1136], [0.0183, 0.0181, 0.0182]),
    ]
)
COMPLEX_BIG_TRANSFORMATIONS_TRAIN = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1145, 0.1144, 0.1134], [0.1694, 0.1675, 0.1684]),
        GaussianNoiseTransform(std=0.001),
    ]
)
SIMPLE_BIG_TRANSFORMATIONS_TRAIN = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1145, 0.1144, 0.1134], [0.1694, 0.1675, 0.1684]),
    ]
)
BIG_TRANSFORMATIONS_TEST = transforms.Compose(
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


def cnn_categorization(model, train_transformations, test_transformations):
    train_dataset = CategorizationDataset(TRAIN_IMG_DIRS, train_transformations)
    test_dataset = CategorizationDataset(TEST_IMG_DIRS, test_transformations)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model, losses, accuracies = train_model(
        model, train_dataset, test_dataset, loss, optimizer, device, scheduler
    )

    test_accuracy = accuracy_dataset(test_dataset, model, device)

    return model, losses, accuracies, test_accuracy


def small_cnn_categorization(TRANSFORMATIONS_TRAIN):
    print("Small CNN for categorization...")

    model = create_small_cnn(CATEGORIES)
    model, losses, accuracies, test_accuracy = cnn_categorization(
        model, TRANSFORMATIONS_TRAIN, SMALL_TRANSFORMATIONS_TEST
    )

    print(f"Test accuracy: {test_accuracy}")
    return model, losses, accuracies, test_accuracy


def big_vgg_categorization(TRANSFORMATIONS_TRAIN):
    print("VGG CNN for categorization...")

    model = create_vgg(CATEGORIES)
    model, losses, accuracies, test_accuracy = cnn_categorization(
        model, TRANSFORMATIONS_TRAIN, BIG_TRANSFORMATIONS_TEST
    )

    print(f"Test accuracy: {test_accuracy}")
    return model, losses, accuracies, test_accuracy


def big_res50_categorization(TRANSFORMATIONS_TRAIN):
    print("RN50 CNN for categorization...")

    model = create_resnet50(CATEGORIES)
    model, losses, accuracies, test_accuracy = cnn_categorization(
        model, TRANSFORMATIONS_TRAIN, BIG_TRANSFORMATIONS_TEST
    )

    print(f"Test accuracy: {test_accuracy}")
    return model, losses, accuracies, test_accuracy


def big_res101_categorization(TRANSFORMATIONS_TRAIN):
    print("RN101 CNN for categorization...")

    model = create_resnet101(CATEGORIES)
    model, losses, accuracies, test_accuracy = cnn_categorization(
        model, TRANSFORMATIONS_TRAIN, BIG_TRANSFORMATIONS_TEST
    )

    print(f"Test accuracy: {test_accuracy}")
    return model, losses, accuracies, test_accuracy



def main(args, model_save_path=None):
    SIZE, TRASFORM = args

    categorization_function = None
    transformation_train = None

    if SIZE == "SMALL":
        categorization_function = small_cnn_categorization

        if TRASFORM == "SIMPLE":
            transformation_train = SIMPLE_SMALL_TRANSFORMATIONS_TRAIN
        elif TRASFORM == "COMPLEX":
            transformation_train = COMPLEX_SMALL_TRANSFORMATIONS_TRAIN
        else:
            print("Invalid transformation.")
            return

    elif SIZE == "VGG" or SIZE == "RES50" or SIZE == "RES101":
        categorization_function = big_cnn_categorization

        if SIZE == "VGG":
            categorization_function = big_vgg_categorization
        elif SIZE == "RES50":
            categorization_function = big_res50_categorization
        elif SIZE == "RES101":
            categorization_function = big_res101_categorization
        else:
            print("Invalid architecture.")
            return
        
        if TRASFORM == "SIMPLE":
            transformation_train = SIMPLE_BIG_TRANSFORMATIONS_TRAIN
        elif TRASFORM == "COMPLEX":
            transformation_train = COMPLEX_BIG_TRANSFORMATIONS_TRAIN
        else:
            print("Invalid transformation.")
            return
    else:
        print("Invalid model size.")
        return

    model, losses, accuracies, test_accuracy = categorization_function(
        transformation_train
    )

    if model_save_path:
        torch.save(model.state_dict(), model_save_path)


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 2:
        print("Invalid number of parameters.")
    else:
        main(args)
