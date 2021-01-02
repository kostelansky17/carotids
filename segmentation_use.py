from os import listdir

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torch import device, load, save
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

from carotids.preprocessing import load_img
from carotids.segmentation.model import Unet

CATEGORIES = 3
DEVICE = device("cpu")
TRANSFORMATIONS_TORCH = Compose(
    [
        Resize((512, 512)),
        ToTensor(),
    ]
)

PATH_TO_TRANS_MODEL = "INSERT_PATH"
PATH_TO_TRANS_DATA = "data_samples/segmentation_samples/transverse"

PATH_TO_LONG_MODEL = "INSERT_PATH"
PATH_TO_LONG_DATA = "data_samples/segmentation_samples/longitudinal"


def segmentation_example_use(path_to_model, path_to_data):
    """Plots segmentation masks of images specified by path_to_data
    created by the model saved in path_to_model.

    Parameters
    ----------
    path_to_model : str
        Path to the model.
    path_to_data : str
        Path to the data.
    """
    model = Unet(CATEGORIES)
    model.load_state_dict(load(path_to_model, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image_names = sorted(listdir(path_to_data))

    for image_name in image_names:
        print(f"Image: {image_name}")
        img = load_img(path_to_data, image_name)
        img_tensor = TRANSFORMATIONS_TORCH(img)
        img_tensor.to(DEVICE)

        mask = model(img_tensor.unsqueeze(0))

        plt.imshow(
            mask.squeeze().detach().cpu().numpy().argmax(0) / CATEGORIES, cmap=cm.gray
        )
        plt.show()


if __name__ == "__main__":
    print("Transverse data...")
    segmentation_example_use(PATH_TO_TRANS_MODEL, PATH_TO_TRANS_DATA)
    print("Longitudinal data...")
    segmentation_example_use(PATH_TO_LONG_MODEL, PATH_TO_LONG_DATA)
