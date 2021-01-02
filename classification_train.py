from torch import cuda, device, save
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)

from carotids.classification.train_model import train_model
from carotids.classification.dataset import ClassificationDataset
from carotids.classification.models import create_resnet50
from carotids.metrics import evaluate_classification_model

TRAIN_IMG_DIRS = {
    0: "INSERT_PATH",
    1: "INSERT_PATH",
    2: "INSERT_PATH",
    3: "INSERT_PATH",
}
VAL_IMG_DIRS = {0: "INSERT_PATH", 1: "INSERT_PATH", 2: "INSERT_PATH", 3: "INSERT_PATH"}
TEST_IMG_DIRS = {0: "INSERT_PATH", 1: "INSERT_PATH", 2: "INSERT_PATH", 3: "INSERT_PATH"}

CLASSES = 4
EPOCHS = 30

TRANSFORMATIONS_TRAIN = Compose(
    [
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        Resize((224, 224)),
        ToTensor(),
    ]
)
TRANSFORMATIONS_TEST = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
    ]
)


def train_classification_model():
    """The approach which has been used for training best classification model
    as defined and described in the thesis.
    """
    torch_device = device("cuda") if cuda.is_available() else device("cpu")

    train_dataset = ClassificationDataset(TRAIN_IMG_DIRS, TRANSFORMATIONS_TRAIN)
    val_dataset = ClassificationDataset(VAL_IMG_DIRS, TRANSFORMATIONS_TEST)
    test_dataset = ClassificationDataset(TEST_IMG_DIRS, TRANSFORMATIONS_TEST)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = create_resnet50(CLASSES)
    model.to(torch_device)

    loss = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.95)
    scheduler = ReduceLROnPlateau(optimizer, patience=3)

    model, losses, accuracies = train_model(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        torch_device,
        scheduler,
        EPOCHS,
    )

    evaluate_classification_model(model, test_loader, loss, torch_device)
    save(model.state_dict(), "classification_model.pt")


if __name__ == "__main__":
    train_classification_model()
