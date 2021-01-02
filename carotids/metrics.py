from torch import device, no_grad, tensor
from torch import max as torch_max
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


def accuracy_torch(output: tensor, label: tensor) -> float:
    """Computes accuracy of a model's output.

    Parameters
    ----------
    output : tensor
        Probabilities predicted by a model.
    label : tensor
        True labels.

    Returns
    -------
    float
        Accuracy of predictions.
    """
    _, predicted = torch_max(output.data, 1)

    return (predicted == label).sum().item() / len(label)


@no_grad()
def accuracy_dataset(dataset: Dataset, model: Module, device: device) -> float:
    """Computes accuracy of a model on a given dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to evaluate model on.
    model : Module
        Model to evaluate.
    device : device
        Device on which is the model.

    Returns
    -------
    float
        Accuracy of a model on a dataset.
    """
    accuracy = 0.0
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    for inputs, labels in dataloader:
        model.eval()

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        accuracy += accuracy_torch(outputs, labels) * inputs.size(0)

    return accuracy / len(dataset)


@no_grad()
def evaluate_classification_model(
    model: Module, dataloader: DataLoader, loss: _Loss, device: device
) -> tuple:
    """Computes loss and accuracy of a model on a given dataloader.

    Parameters
    ----------
    model : Module
        Model to evaluate.
    dataloader : DataLoader
        Dataloader to evaluate model on.
    loss : _Loss
        A loss function to use.
    device : device
        Device on which is the model.

    Returns
    -------
    tuple
        Mean loss and accuracy of a model on a dataloader.
    """
    model.eval()
    model_loss = 0.0
    model_acc = 0

    data_size = len(dataloader.dataset)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        l = loss(outputs, labels)

        model_loss += l.item() * inputs.size(0)
        model_acc += accuracy_torch(outputs, labels) * inputs.size(0)

    return model_loss / data_size, model_acc / data_size


def iou(labels: tensor, outputs: tensor, treshold: float) -> int:
    """Computes a number of predictions with IoU highier than the treshold
    against true bounding boxes.

    Parameters
    ----------
    labels : tensor
        True bounding boxes.
    outputs : tensor
        Predicted bounding boxes.
    treshold : float
        Treshold for IoU.

    Returns
    -------
    int
        Number of predictions with IoU highier than a treshold.
    """
    size = len(labels)

    correct = 0
    for i in range(size):
        if outputs[i, 0] >= outputs[i, 2] or outputs[i, 1] >= outputs[i, 3]:
            continue

        x_0 = max(labels[i, 0], outputs[i, 0])
        y_0 = max(labels[i, 1], outputs[i, 1])

        x_1 = min(labels[i, 2], outputs[i, 2])
        y_1 = min(labels[i, 3], outputs[i, 3])

        if x_1 < x_0 or y_1 < y_0:
            continue

        intersection_area = (x_1 - x_0) * (y_1 - y_0)

        label_area = (labels[i, 2] - labels[i, 0]) * (labels[i, 3] - labels[i, 1])
        output_area = (outputs[i, 2] - outputs[i, 0]) * (outputs[i, 3] - outputs[i, 1])

        iou = intersection_area / float(label_area + output_area - intersection_area)

        if iou >= treshold:
            correct += 1

    return correct


@no_grad()
def evaluate_dataset_iou_frcnn(
    model: Module, data_loader: DataLoader, device: device, treshold: float = 0.85
) -> tuple:
    """Computes a number of predictions of a Faster R-CNNmodel on a dataloader
    with IoU highier than the treshold, and number of images on which is found
    zero predicted objects, one predicted object or many predicted object.

    Parameters
    ----------
    model : Module
        Model to be evaluated.
    data_loader : DataLoader
        Dataloader to evaluate the model on.
    device : device
        Device on which is the model.
    treshold : float
        Treshold for IoU.

    Returns
    -------
    tuple
        Number of predictions with IoU highier than a treshold, number of images
        on which is found zero predicted objects, one predicted object and many
        predicted object.
    """
    zero_predictions = 0
    one_prediction = 0
    many_predictions = 0
    acc = 0

    model.eval()

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        outputs = model(images)

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        for i in range(len(outputs)):
            if len(outputs[i]["boxes"]) > 0:
                many_predictions += 1
                acc += iou(
                    targets[i]["boxes"][0].unsqueeze(0),
                    outputs[i]["boxes"][0].unsqueeze(0),
                    treshold,
                )
            elif len(outputs[i]["boxes"]) == 1:
                one_prediction += 1
                acc += iou(
                    targets[i]["boxes"][0].unsqueeze(0),
                    outputs[i]["boxes"][0].unsqueeze(0),
                    treshold,
                )
            else:
                zero_predictions += 1

    return acc, zero_predictions, one_prediction, many_predictions


@no_grad()
def evaluate_dataset_iou_resnet(
    model: Module, dataset: Dataset, device: device, treshold: float = 0.85
) -> float:
    """Computes a number of predictions of a ResNet on a dataset with IoU
    highier than the treshold.

    Parameters
    ----------
    model : Module
        Model to be evaluated.
    dataset : Dataset
        Dataset to evaluate the model on.
    device : device
        Device on which is the model.
    treshold : float
        Treshold for IoU.

    Returns
    -------
    int
        Percentage of predictions with IoU highier than a treshold.
    """
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model.eval()

    s = 0
    for inputs, labels in data_loader:
        output = model(inputs.to(device)).cpu()
        s += iou(labels.int(), output.int(), 0.85)

    return s / len(dataset)
