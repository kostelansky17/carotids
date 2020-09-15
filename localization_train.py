import copy
import sys

import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torch.utils.data import DataLoader
from torchvision import transforms

from carotids.localization.detection import utils
from carotids.localization.detection.engine import train_one_epoch
from carotids.localization.frcnn_dataset import FastCarotidDataset
from carotids.localization.models import create_faster_rcnn, create_resnet_model
from carotids.localization.resnet_dataset import ResnetCarotidDataset
from carotids.metrics import evaluate_dataset_iou_resnet, evaluate_dataset_iou_frcnn
from carotids.train_model import train_model


TRAIN_IMGS_PATH = "drive/My Drive/cartroids/data/train/img/"
TRAIN_LABELS_PATH = "drive/My Drive/cartroids/data/train/txt/"
VALIDATION_IMGS_PATH = "drive/My Drive/cartroids/data/validation/img/"
VALIDATION_LABELS_PATH = "drive/My Drive/cartroids/data/validation/txt/"
TEST_IMGS_PATH = "drive/My Drive/cartroids/data/test/img/"
TEST_LABELS_PATH = "drive/My Drive/cartroids/data/test/txt/"


TRANSFORMATIONS_ONE = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1323, 0.1323, 0.1323], [0.1621, 0.1621, 0.1621]),
    ]
)

TRANSFORMATIONS_TWO = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class ModelTypes(Enum):
    RESNET = "RESNET"
    FRCNN = "FRCNN"


def train_renset_localization():
    train_dataset = ResnetCarotidDataset(
        TRAIN_IMGS_PATH, TRAIN_LABELS_PATH, TRANSFORMATIONS_ONE
    )
    val_dataset = ResnetCarotidDataset(
        VALIDATION_IMGS_PATH, VALIDATION_LABELS_PATH, TRANSFORMATIONS_ONE
    )
    test_dataset = ResnetCarotidDataset(
        TEST_IMGS_PATH, TEST_LABELS_PATH, TRANSFORMATIONS_ONE
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
    train_dataset = FastCarotidDataset(
        TRAIN_IMGS_PATH, TRAIN_LABELS_PATH, TRANSFORMATIONS_TWO
    )
    val_dataset = FastCarotidDataset(
        VALIDATION_IMGS_PATH, VALIDATION_LABELS_PATH, TRANSFORMATIONS_TWO
    )
    test_dataset = FastCarotidDataset(
        TEST_IMGS_PATH, TEST_LABELS_PATH, TRANSFORMATIONS_TWO
    )

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn
    )

    model = create_faster_rcnn()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.001, momentum=0.9)
    num_epochs = 20

    best_model = None
    best_eval = 0.0

    for epoch in range(33):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        eval = evaluate_dataset_iou_frcnn(model, val_loader, device=device) / len(
            val_dataset
        )

        if eval > best_eval:
            best_eval = eval
            best_model = copy.deepcopy(model)
        print(f"Val eval: {eval}")

    return best_model


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
