import sys

import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import transforms

from carotids.localization.frcnn_dataset import FastCarotidDataset
from carotids.localization.models import create_faster_rcnn, create_resnet_model
from carotids.localization.resnet_dataset import ResnetCarotidDataset
from carotids.metrics import evaluate_dataset_iou_resnet

TRAIN_IMGS_PATH = "drive/My Drive/cartroids/data/train/img/"
TRAIN_LABELS_PATH = "drive/My Drive/cartroids/data/train/txt/"
VALIDATION_IMGS_PATH = "drive/My Drive/cartroids/data/validation/img/"
VALIDATION_LABELS_PATH = "drive/My Drive/cartroids/data/validation/txt/"
TEST_IMGS_PATH = "drive/My Drive/cartroids/data/test/img/"
TEST_LABELS_PATH = "drive/My Drive/cartroids/data/test/txt/"


TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1323, 0.1323, 0.1323], [0.1621, 0.1621, 0.1621]),
    ]
)


class ModelTypes(Enum):
    RESNET = "RESNET"
    FRCNN = "FRCNN"


def train_renset_localization():
    train_dataset = ResnetCarotidDataset(
        TRAIN_IMGS_PATH, TRAIN_LABELS_PATH, TRANSFORMATIONS
    )
    val_dataset = ResnetCarotidDataset(
        VALIDATION_IMGS_PATH, VALIDATION_LABELS_PATH, TRANSFORMATIONS
    )
    test_dataset = ResnetCarotidDataset(
        TEST_IMGS_PATH, TEST_LABELS_PATH, TRANSFORMATIONS
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_resnet_model()
    model = model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model, losses, accuracies = train_model(
        model, train_dataset, test_dataset, loss, optimizer, device, scheduler
    )

    test_iou = evaluate_dataset_iou_resnet(model, test_dataset, device)

    return model, test_iou


def train_frcnn_localization():
    pass


def main(model_type):
    if model_type == ModelTypes.RESNET:
        train_renset_localization()
    elif model_type == ModelTypes.FRCNN:
        train_renset_localization()


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 1:
        print("Invalid number of parameters.")
    else:
        main(args)
