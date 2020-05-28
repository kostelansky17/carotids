import torch
import torch.nn as nn
from torch.optim import lr_scheduler, SGD
from torchvision import transforms
from torchvision import models, transforms

from carotids.categorization.categorization_cnn import create_small_cnn, create_vgg
from carotids.categorization.categorization_dataset import CategorizationDataset
from carotids.train_model import train_model


IMG_DIRS = {
   0: "/home/martin/Documents/cartroids/data/categorization/train/diff",
   1: "/home/martin/Documents/cartroids/data/categorization/train/long",
   2: "/home/martin/Documents/cartroids/data/categorization/train/trav",
}

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


def main():
    dataset = CategorizationDataset(IMG_DIRS, SMALL_TRANSFORMATIONS)
    model = create_small_cnn(CATEGORIES)

    #dataset = CategorizationDataset(IMG_DIRS, BIG_TRANSFORMATIONS)
    #model = create_vgg(CATEGORIES)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model, losses, accuracies = train_model(model, dataset, loss, optimizer, device, scheduler)


if __name__ == "__main__":
    main()