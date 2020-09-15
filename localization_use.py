import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from carotids.localization.frcnn_dataset import FastCarotidDatasetEval
from carotids.localization.models import create_faster_rcnn

TRANSFORMATIONS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

PATH_TO_DATA = "/home/martin/Documents/cartroids/data_samples/localization_samples"
PATH_TO_MODEL = "/home/martin/Documents/cartroids/models/localization_model.pth"


def categorization_example_use():
    model = create_faster_rcnn()
    model.load_state_dict(torch.load(PATH_TO_MODEL))
    model.eval()

    dataset = FastCarotidDatasetEval(PATH_TO_DATA, TRANSFORMATIONS)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image_tensor, image_name in loader:
        outputs = model(image_tensor)
        print(
            "Image {} : Coordinates:{}, Score:{:.2f}%".format(
                image_name[0],
                outputs[0]["boxes"][0].detach().numpy(),
                outputs[0]["scores"][0],
            )
        )


if __name__ == "__main__":
    categorization_example_use()
