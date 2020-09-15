import os

import torch
import torch.nn as nn
from torchvision import transforms

from carotids.preprocessing import load_img
from carotids.categorization.categorization_cnn import create_resnet50

CATEGORIES = 3
TRANSFORMATIONS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.1145, 0.1144, 0.1134], [0.1694, 0.1675, 0.1684]),
    ]
)

PATH_TO_DATA = "/home/martin/Documents/cartroids/data_samples/categorization_samples/"
PATH_TO_MODEL = "/home/martin/Documents/cartroids/models/categorization_model.pth"


def categorization_example_use():
    model = create_resnet50(CATEGORIES)
    model.load_state_dict(torch.load(PATH_TO_MODEL))
    model.eval()

    image_names = os.listdir(PATH_TO_DATA)
    softmax = nn.Softmax(dim=1)

    for image_name in image_names:
        img = load_img(PATH_TO_DATA, image_name)
        img_tensor = TRANSFORMATIONS(img)
        predictions = model(img_tensor.unsqueeze(0))
        probabs = softmax(predictions)
        print(
            "Image {} : Long:{:.2f}%, Trav:{:.2f}%, Diff:{:.2f}%".format(
                image_name, probabs[0][0], probabs[0][1], probabs[0][2]
            )
        )


if __name__ == "__main__":
    categorization_example_use()
