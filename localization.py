import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from carotids.preprocessing import load_position_labels, load_imgs_dir


IMGS_PATH = "drive/My Drive/cartroids/data/img/"
LABELS_PATH = "drive/My Drive/cartroids/data/txt/"

TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class ResnetCartroidDataset(Dataset):
    def __init__(self, imgs_path, labels_path, transformations):
        self.data = load_imgs_dir(imgs_path)
        self.labels = load_position_labels(labels_path)
        self.transformations = transformations

    def __getitem__(self, index):
        label_tensor = torch.from_numpy(self.labels[index])
        return self.transformations(self.data[index]).double(), label_tensor.double()[0]

    def __len__(self):
        return len(self.data)


def create_resnet_model(arch="resnet50", pretrained=True):
    model = torch.hub.load("pytorch/vision:v0.5.0", arch, pretrained=pretrained)
    fc_in_size = model.fc.in_features
    model.fc = nn.Linear(fc_in_size, 1)


def create_dataloaders(dataset, batch_size=5, train_split=0.9):
    trainsize = int(len(dataset) * train_split)
    valsize = len(dataset) - trainsize
    trainset, valset = random_split(dataset, [trainsize, valsize])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    return {"train": train_loader, "val": val_loader}


def main():
    dataset = ResnetCartroidDataset(IMGS_PATH, LABELS_PATH, TRANSFORMATIONS)
    dataloaders = create_dataloaders(dataset)


if __name__ == "__main__":
    main()
