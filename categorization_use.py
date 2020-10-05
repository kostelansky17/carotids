from os import listdir

from torch import device, load, no_grad
from torch.nn import Softmax
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from carotids.preprocessing import load_img
from carotids.categorization.models import create_vgg

CATEGORIES = 3
DEVICE = device("cpu")
TRANSFORMATIONS = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.1145, 0.1144, 0.1134], [0.1694, 0.1675, 0.1684]),
    ]
)

PATH_TO_DATA = "FILL_ME"
PATH_TO_MODEL = "FILL_ME"


@no_grad()
def categorization_example_use() -> None:
    """Example usage of categorization model. Load model from path selected by 
    parameter PATH_TO_MODEL. Evaluates the images in the folder specified by 
    the PATH_TO_DATA parameter. Prints name of the file and probabilities for
    each class.
    """
    model = create_vgg(CATEGORIES)
    model.load_state_dict(load(PATH_TO_MODEL, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image_names = listdir(PATH_TO_DATA)
    softmax = Softmax(dim=1)

    for image_name in image_names:
        img = load_img(PATH_TO_DATA, image_name)
        img_tensor = TRANSFORMATIONS(img)
        img_tensor = img_tensor.to(DEVICE)

        predictions = model(img_tensor.unsqueeze(0))
        probabs = softmax(predictions)
        print(
            "Image {} : Long:{:.2f}%, Trav:{:.2f}%, Diff:{:.2f}%".format(
                image_name, probabs[0][0], probabs[0][1], probabs[0][2]
            )
        )


if __name__ == "__main__":
    categorization_example_use()
