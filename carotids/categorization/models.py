from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, Sequential
from torchvision import models


def create_small_cnn(categories: int) -> Sequential:
    """Creates a convolutional neural network.

    Parameters
    ----------
    categories : int
        The number of categories to predict.

    Returns
    -------
    Sequential
        Returns small convolutional neural network.
    """
    model = Sequential(
        Conv2d(3, 4, 5),
        MaxPool2d(2),
        Conv2d(4, 8, 5),
        MaxPool2d(2),
        Flatten(),
        Linear(4 * 4 * 8, categories),
    )

    return model


def create_vgg(
    categories: int, pretrained: bool = True, all_layers: bool = True
) -> Module:
    """Creates the VGG16 neural network.

    Parameters
    ----------
    categories : int
        The number of categories to predict.
    pretrained : bool
        Flag to create a pretrained model on the ImageNet dataset.
    all_layers : bool
        Flag to set the requires_grad parameter in all layers.
    
    Returns
    -------
    Module
        Returns VGG16 model.
    """
    model = models.vgg16(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = all_layers

    in_features = model.classifier[6].in_features
    model.classifier[6] = Linear(in_features, categories)

    return model


def create_resnet50(
    categories, pretrained: bool = True, all_layers: bool = True
) -> Module:
    """Creates ResNet50 neural network.

    Parameters
    ----------
    categories : int
        The number of categories to predict.
    pretrained : bool
        Flag to create a pretrained model on the ImageNet dataset.
    all_layers : bool
        Flag to set the requires_grad parameter in all layers.
    
    Returns
    -------
    Module
        Returns ResNet50 model.
    """
    model = models.resnet50(pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = all_layers

    in_features = model.fc.in_features
    model.fc = Linear(in_features, categories)

    return model
