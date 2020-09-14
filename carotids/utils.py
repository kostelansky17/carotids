import numpy as np
import torch
from torch.utils.data import DataLoader, random_split


class GaussianNoiseTransform(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


def compute_mean_image_dataloader(dataloader):
    mean = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)

    return mean


def compute_std_image_dataloader(dataloader, mean):
    var = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(dataloader.dataset) * 224 * 224))

    return std


def compute_normalization_image_dataloader(dataloader):
    mean = compute_mean_image_dataloader(dataloader)
    std = compute_std_image_dataloader(dataloader, mean)

    return mean, std


def train_val_split(dataset, val_split=0.1, batch_size=64):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    torch.manual_seed(17)
    trainset, valset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return train_loader, train_size, val_loader, val_size


def recompute_labels(image, x0, y0, x1, y1, target_size=(244, 244)):
    if len(np.asarray(image).shape) == 2:
        img_height, img_width = np.asarray(image).shape
    else:
        img_height, img_width, _ = np.asarray(image).shape

    target_height, target_width = target_size

    x0, x1 = x0 * (img_width / target_width), x1 * (img_width / target_width)
    y0, y1 = y0 * (img_height / target_height), y1 * (img_height / target_height)

    return torch.FloatTensor([x0, y0, x1, y1])


def generate_boxes(w, h, x1, y1, x2, y2):
    boxes = []

    for _ in range(7):
        selection = np.random.randint(0, 4, 1)[0]
        if selection == 0:
            xa = np.random.randint(0, x1, 1)[0]
            ya = np.random.randint(0, h, 1)[0]

            xb = np.random.randint(xa, x1, 1)[0]
            yb = np.random.randint(ya, h, 1)[0]

        elif selection == 1:
            xa = np.random.randint(0, w, 1)[0]
            ya = np.random.randint(y2, h, 1)[0]

            xb = np.random.randint(xa, w, 1)[0]
            yb = np.random.randint(ya, h, 1)[0]

        elif selection == 2:
            xa = np.random.randint(x2, w, 1)[0]
            ya = np.random.randint(0, h, 1)[0]

            xb = np.random.randint(xa, w, 1)[0]
            yb = np.random.randint(ya, h, 1)[0]
        else:
            xa = np.random.randint(0, w, 1)[0]
            ya = np.random.randint(0, y1, 1)[0]

            xb = np.random.randint(xa, w, 1)[0]
            yb = np.random.randint(ya, y1, 1)[0]

        boxes.append([xa, ya, xb, yb])

    return boxes
