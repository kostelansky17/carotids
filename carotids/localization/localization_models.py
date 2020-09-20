from torch.hub import load
from torch.nn import Linear, Module
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_resnet_model(
    arch: str = "resnet50", pretrained: bool = True, all_layers: bool = True
) -> Module:
    """Creates ResNet neural network for localization.

    Parameters
    ----------
    arch : str
        Selected ResNet architecture.
    pretrained : bool
        Flag to create a pretrained model on the ImageNet dataset.
    all_layers : bool
        Flag to set the requires_grad parameter in all layers.
    
    Returns
    -------
    Module
        Returns ResNet model.
    """
    model = load("pytorch/vision:v0.5.0", arch, pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = all_layers

    fc_in_size = model.fc.in_features
    model.fc = Linear(fc_in_size, 4)

    return model.double()


def create_faster_rcnn(
    pretrained: bool = False, trainable_backbone_layers: int = 3
) -> FasterRCNN:
    """Creates Faster R-CNN model.

    Parameters
    ----------
    pretrained : bool
        Flag to create a pretrained model on the ImageNet dataset.
    trainable_backbone_layers : int
        Number of backbone layers to train.

    Returns
    -------
    FasterRCNN
        Returns Faster R-CNN model.
    """
    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained, trainable_backbone_layers=trainable_backbone_layers
    )
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
