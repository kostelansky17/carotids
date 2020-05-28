import copy

import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import transforms

from categorization_dataset import CategorizationDataset, IMG_DIRS
from train_model import train_model

CATEGORIES = 3
SMALL_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.1147, 0.1146, 0.1136], [0.0183, 0.0181, 0.0182])
])
BIG_TRANSFORMATIONS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.1145, 0.1144, 0.1134], [0.1694, 0.1675, 0.1684])
])


def create_small_cnn(categories):
    model = nn.Sequential(
        nn.Conv2d(3, 4, 5),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 5),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(4 * 4 * 8, categories),
    )

    return model

def create_vgg(categories, pretrained=True, all_layers=True):
    model = torchvision.models.vgg11(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = all_layers

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, categories)

    return model


def main():
    dataset = CategorizationDataset(IMG_DIRS, SMALL_TRANSFORMATIONS)
    model = create_small_cnn(CATEGORIES)

    #dataset = CategorizationDataset(IMG_DIRS, BIG_TRANSFORMATIONS)
    #model = create_small_cnn(CATEGORIES)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model, losses, accuracies = train_model(model, dataset, loss, optimizer, device, scheduler)


if __name__ == "__main__":
    main()
