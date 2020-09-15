import numpy as np
import torch
from torch.utils.data import DataLoader


def accuracy_torch(output, label):
    _, predicted = torch.max(output.data, 1)

    return (predicted == label).sum().item() / len(label)


def accuracy_np(output, label):
    return np.mean(output == label)


def accuracy_dataset(dataset, model, device):
    accuracy = 0.0
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    for inputs, labels in dataloader:
        model.eval()

        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            accuracy += accuracy_torch(outputs, labels) * inputs.size(0)

    return accuracy / len(dataset)


def iou(labels, outputs, treshold):
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


@torch.no_grad()
def evaluate_dataset_iou_frcnn(model, data_loader, device):
    acc = 0
    model.eval()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        outputs = model(images)

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]

        if len(outputs[0]["boxes"]):
            acc += iou(
                targets[0]["boxes"][0].unsqueeze(0).int(),
                outputs[0]["boxes"][0].unsqueeze(0).int(),
                0.6,
            )

        if len(outputs) > 1:
            if len(outputs[1]["boxes"]):
                acc += iou(
                    targets[1]["boxes"][0].unsqueeze(0).int(),
                    outputs[1]["boxes"][0].unsqueeze(0).int(),
                    0.6,
                )

    return acc


def evaluate_dataset_iou_resnet(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    s = 0
    for inputs, labels in loader:
        with torch.no_grad():
            output = model(inputs.to(device)).cpu()

        s += iou(labels.int(), output.int(), 0.85)

    return s / len(dataset)
