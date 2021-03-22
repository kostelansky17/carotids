from torch import device, load, no_grad
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from carotids.localization.frcnn_dataset import FastCarotidDatasetEval
from carotids.localization.models import create_faster_rcnn

DEVICE = device("cpu")
TRANSFORMATIONS = Compose([ToTensor()])

PATH_TO_TRANS_MODEL = "INSERT_PATH"
PATH_TO_TRANS_DATA = "data_samples/localization_samples/transverse"

PATH_TO_LONG_MODEL = "INSERT_PATH"
PATH_TO_LONG_DATA = "data_samples/localization_samples/longitudinal"


@no_grad()
def localization_example_use(path_to_model: str, path_to_data: str) -> None:
    """Example usage of localization model. Load model from path selected by
    parameter path_to_model. Evaluates the images in the folder specified by
    the path_to_data parameter. Prints name of the file and coordinates of the
    most probable carotid on the image with its score.

    Parameters
    ----------
    path_to_model : str
        Path to the model.
    path_to_data : str
        Path to the data.
    """
    model = create_faster_rcnn()
    model.load_state_dict(load(path_to_model, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dataset = FastCarotidDatasetEval(path_to_data, TRANSFORMATIONS)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for image_tensor, image_name in loader:
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        print(
            "Image {} : Coordinates:{}, Score:{:.2f}%".format(
                image_name[0],
                outputs[0]["boxes"][0].detach().numpy(),
                outputs[0]["scores"][0],
            )
        )


if __name__ == "__main__":
    print("Transverse data:")
    localization_example_use(PATH_TO_TRANS_MODEL, PATH_TO_TRANS_DATA)
    print("Longitudinal data:")
    localization_example_use(PATH_TO_LONG_MODEL, PATH_TO_LONG_DATA)
