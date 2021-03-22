import copy

from torch import cuda, device, save
from torch.nn import Module
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from carotids.localization.evaluation import eval_one_epoch
from carotids.localization.frcnn_dataset import (
    FastCarotidDatasetANTIQUE,
    FastCarotidDatasetSPLab,
)
from carotids.localization.models import create_faster_rcnn
from carotids.localization.train_model import train_one_epoch
from carotids.localization.transformations import (
    LocCompose,
    LocCrop,
    LocRandomHorizontalFlip,
    LocRandomVerticalFlip,
    LocReshape,
)
from carotids.localization.utils import collate_fn
from carotids.utils import split_dataset


EPOCHS = 40

# transformations
TRANSFORMATIONS_TORCH = Compose(
    [
        ToTensor(),
    ]
)

TRANSFORMATIONS_CUSTOM = LocCompose(
    [
        LocRandomHorizontalFlip(0.5),
        LocRandomVerticalFlip(0.5),
        LocCrop(0.1),
        LocReshape(0.25, 0.8, 1.2),
    ]
)

TRANSFORMATIONS_CUSTOM_SMPL = LocCompose([])


# SPLab DATA
SPLab_IMGS_PATH = "INSERT_PATH"
SPLab_LABELS_PATH = "INSERT_PATH"

SPLab_dataset = FastCarotidDatasetSPLab(
    SPLab_IMGS_PATH, SPLab_LABELS_PATH, TRANSFORMATIONS_CUSTOM, TRANSFORMATIONS_TORCH
)
SPLab_dataset_train, _, SPLab_dataset_val, _ = split_dataset(SPLab_dataset, 0.2, 2)
SPLab_train_loader = DataLoader(
    SPLab_dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn
)
SPLab_val_loader = DataLoader(
    SPLab_dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn
)


# ANTIQUE TRANSVERSAL DATA - TRAIN
ANTIQUE_TRANS_TRAIN_IMGS_PATH = "INSERT_PATH"
ANTIQUE_TRANS_TRAIN_LABELS_PATH = "INSERT_PATH"

ANTIQUE_trans_dataset_train = FastCarotidDatasetANTIQUE(
    ANTIQUE_TRANS_TRAIN_IMGS_PATH,
    ANTIQUE_TRANS_TRAIN_LABELS_PATH,
    TRANSFORMATIONS_CUSTOM,
    TRANSFORMATIONS_TORCH,
)
ANTIQUE_trans_train_loader = DataLoader(
    ANTIQUE_trans_dataset_train, batch_size=2, shuffle=False, collate_fn=collate_fn
)

# ANTIQUE TRANSVERSAL DATA - VAL
ANTIQUE_TRANS_VAL_IMGS_PATH = "INSERT_PATH"
ANTIQUE_TRANS_VAL_LABELS_PATH = "INSERT_PATH"

ANTIQUE_trans_dataset_val = FastCarotidDatasetANTIQUE(
    ANTIQUE_TRANS_VAL_IMGS_PATH,
    ANTIQUE_TRANS_VAL_LABELS_PATH,
    TRANSFORMATIONS_CUSTOM_SMPL,
    TRANSFORMATIONS_TORCH,
)
ANTIQUE_trans_val_loader = DataLoader(
    ANTIQUE_trans_dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn
)

# ANTIQUE LONGITUDINAL DATA - TRAIN
ANTIQUE_LONG_TRAIN_IMGS_PATH = "INSERT_PATH"
ANTIQUE_LONG_TRAIN_LABELS_PATH = "INSERT_PATH"

ANTIQUE_long_dataset_train = FastCarotidDatasetANTIQUE(
    ANTIQUE_LONG_TRAIN_IMGS_PATH,
    ANTIQUE_LONG_TRAIN_LABELS_PATH,
    TRANSFORMATIONS_CUSTOM_SMPL,
    TRANSFORMATIONS_TORCH,
)
ANTIQUE_long_train_loader = DataLoader(
    ANTIQUE_long_dataset_train, batch_size=2, shuffle=False, collate_fn=collate_fn
)

# ANTIQUE LONGITUDINAL DATA - VAL
ANTIQUE_LONG_VAL_IMGS_PATH = "INSERT_PATH"
ANTIQUE_LONG_VAL_LABELS_PATH = "INSERT_PATH"

ANTIQUE_long_dataset_val = FastCarotidDatasetANTIQUE(
    ANTIQUE_LONG_VAL_IMGS_PATH,
    ANTIQUE_LONG_VAL_LABELS_PATH,
    TRANSFORMATIONS_CUSTOM_SMPL,
    TRANSFORMATIONS_TORCH,
)
ANTIQUE_long_val_loader = DataLoader(
    ANTIQUE_long_dataset_val, batch_size=2, shuffle=False, collate_fn=collate_fn
)


def train_transverse_fasterrcnn_model() -> Module:
    """The approach which has been used for the best transverse Faster R-CNN
    as defined and described in the thesis.

    Returns
    -------
    Module
        Trained model.
    """
    model = create_faster_rcnn(True)
    torch_device = device("cuda") if cuda.is_available() else device("cpu")

    model.to(torch_device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=0.0001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [5, 15, 25])

    best_model = None
    best_val_loss = 10 ** 8

    print("Training on the SPLab data...")
    for epoch in range(EPOCHS):
        train_logger = train_one_epoch(
            model, optimizer, SPLab_train_loader, torch_device, epoch, print_freq=200
        )
        val_logger = eval_one_epoch(
            model, SPLab_val_loader, torch_device, print_freq=200
        )

        print(
            f"Epoch: {epoch}, SPLab, Train loss: {train_logger.loss.avg}, Train loss_classifier: {train_logger.loss_classifier.avg}, Train loss_box_reg: {train_logger.loss_box_reg.avg}, Train loss_objectness: {train_logger.loss_objectness.avg}"
        )
        print(
            f"Epoch: {epoch}, SPLab, Val. loss: {val_logger.loss.avg}, Val. loss_classifier: {val_logger.loss_classifier.avg}, Val. loss_box_reg: {val_logger.loss_box_reg.avg}, Val. loss_objectness: {val_logger.loss_objectness.avg}"
        )

        scheduler.step()

        if val_logger.loss.avg < best_val_loss:
            best_val_loss = val_logger.loss.avg
            best_model = copy.deepcopy(model)

    model.load_state_dict(best_model.state_dict())

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=0.0001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [5, 15, 25])

    best_model = None
    best_val_loss = 10 ** 8

    print("Finetuning on the ANTIQUE data...")
    for epoch in range(EPOCHS):
        train_logger = train_one_epoch(
            model,
            optimizer,
            ANTIQUE_trans_train_loader,
            torch_device,
            epoch,
            print_freq=200,
        )
        val_logger = eval_one_epoch(
            model, ANTIQUE_trans_train_loader, torch_device, print_freq=200
        )

        print(
            f"Epoch: {epoch}, ANTIQUE, Train loss: {train_logger.loss.avg}, Train loss_classifier: {train_logger.loss_classifier.avg}, Train loss_box_reg: {train_logger.loss_box_reg.avg}, Train loss_objectness: {train_logger.loss_objectness.avg}"
        )
        print(
            f"Epoch: {epoch}, ANTIQUE, Val. loss: {val_logger.loss.avg}, Val. loss_classifier: {val_logger.loss_classifier.avg}, Val. loss_box_reg: {val_logger.loss_box_reg.avg}, Val. loss_objectness: {val_logger.loss_objectness.avg}"
        )

        scheduler.step()

        if val_logger.loss.avg < best_val_loss:
            best_val_loss = val_logger.loss.avg
            best_model = copy.deepcopy(model)

    save(best_model.state_dict(), "transverse_localization_model.pt")


def train_longitudinal_fasterrcnn_model() -> Module:
    """The approach which has been used for the best longitudinal Faster R-CNN
    as defined and described in the thesis.

    Returns
    -------
    Module
        Trained model.
    """
    model = create_faster_rcnn(True)
    torch_device = device("cuda") if cuda.is_available() else device("cpu")

    model.to(torch_device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=0.0001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [5, 15, 25])

    best_model = None
    best_val_loss = 10 ** 8

    for epoch in range(EPOCHS):
        train_logger = train_one_epoch(
            model,
            optimizer,
            ANTIQUE_long_train_loader,
            torch_device,
            epoch,
            print_freq=200,
        )
        val_logger = eval_one_epoch(
            model, ANTIQUE_long_val_loader, torch_device, print_freq=200
        )

        print(
            f"Epoch: {epoch}, ANTIQUE, Train loss: {train_logger.loss.avg}, Train loss_classifier: {train_logger.loss_classifier.avg}, Train loss_box_reg: {train_logger.loss_box_reg.avg}, Train loss_objectness: {train_logger.loss_objectness.avg}"
        )
        print(
            f"Epoch: {epoch}, ANTIQUE, Val. loss: {val_logger.loss.avg}, Val. loss_classifier: {val_logger.loss_classifier.avg}, Val. loss_box_reg: {val_logger.loss_box_reg.avg}, Val. loss_objectness: {val_logger.loss_objectness.avg}"
        )

        scheduler.step()

        if val_logger.loss.avg < best_val_loss:
            best_val_loss = val_logger.loss.avg
            best_model = copy.deepcopy(model)

    save(best_model.state_dict(), "longitudinal_localization_model.pt")


if __name__ == "__main__":
    train_transverse_fasterrcnn_model()
    train_longitudinal_fasterrcnn_model()
