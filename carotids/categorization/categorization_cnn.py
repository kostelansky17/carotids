import torch.nn as nn
from torchvision import models


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
    model = models.vgg11(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = all_layers

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, categories)

    return model


def create_resnet50(categories, pretrained=True, all_layers=True):
    model =  models.resnet50(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = all_layers

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, categories)

    return model


def create_resnet101(categories, pretrained=True, all_layers=True):
    model =  models.resnet101(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = all_layers

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, categories)

    return model
