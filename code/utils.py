import torch
from torch.utils.data import DataLoader, random_split


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
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(dataloader.dataset)*224*224))

    return std


def compute_normalization_image_dataloader(dataloader):
    mean = compute_mean_image_dataloader(dataloader)
    std = compute_std_image_dataloader(dataloader, mean)

    return mean, std


def train_val_split(dataset, val_split=0.1, batch_size=64):
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return train_loader, train_size, val_loader, val_size
