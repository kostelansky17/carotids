import torch
from torch import nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_resnet_model(arch="resnet50", pretrained=True, all_layers=True):
    model = torch.hub.load("pytorch/vision:v0.5.0", arch, pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = all_layers

    fc_in_size = model.fc.in_features
    model.fc = nn.Linear(fc_in_size, 4)

    return model.double()


def create_faster_rcnn(pretrained=False):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
