import torch
from torch import nn


def create_resnet_model(caegories, arch="resnet50", pretrained=True):
    model = torch.hub.load("pytorch/vision:v0.5.0", arch, pretrained=pretrained)
    fc_in_size = model.fc.in_features
    model.fc = nn.Linear(fc_in_size, caegories)

    return model
