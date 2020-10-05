from torch import device, load, no_grad
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from carotids.localization.frcnn_dataset import FastCarotidDatasetEval
from carotids.localization.models import create_faster_rcnn

DEVICE = device("cpu")
TRANSFORMATIONS = Compose(
    [ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)

PATH_TO_DATA = "/home/martin/Documents/cartroids/data_samples/localization_samples"
PATH_TO_MODEL = "/home/martin/Documents/cartroids/models/localization_model.pth"


@no_grad()
def localization_example_use() -> None:
    """Example usage of localization model. Load model from path selected by 
    parameter PATH_TO_MODEL. Evaluates the images in the folder specified by 
    the PATH_TO_DATA parameter. Prints name of the file and coordinates of the
    most probable carotid on the image with its score.
    """
    model = create_faster_rcnn()
    model.load_state_dict(load(PATH_TO_MODEL, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dataset = FastCarotidDatasetEval(PATH_TO_DATA, TRANSFORMATIONS)
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
    localization_example_use()
