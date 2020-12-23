from os import listdir

from torch import device, load, no_grad
from torch.nn import Softmax
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from carotids.preprocessing import load_img
from carotids.categorization.models import create_resnet50

CATEGORIES = 4
DEVICE = device("cpu")
TRANSFORMATIONS = Compose(
    [
        Resize((224, 224)),
        ToTensor(),
    ]
)

PATH_TO_DATA = "TO_FILL"
PATH_TO_MODEL = "TO_FILL"

    
@no_grad()
def categorization_example_use() -> None:
    """Example usage of categorization model. Load model from path selected by 
    parameter PATH_TO_MODEL. Evaluates the images in the folder specified by 
    the PATH_TO_DATA parameter. Prints name of the file and probabilities for
    each class.
    """
    model = create_resnet50(CATEGORIES)
    model.load_state_dict(load(PATH_TO_MODEL, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    image_names = sorted(listdir(PATH_TO_DATA))
    softmax = Softmax(dim=1)

    for image_name in image_names:
        img = load_img(PATH_TO_DATA, image_name)
        img_tensor = TRANSFORMATIONS(img)
        img_tensor = img_tensor.to(DEVICE)

        predictions = model(img_tensor.unsqueeze(0))
        probabs = softmax(predictions)
        print(
            "Image {} : Long:{:.2f}%, Trav:{:.2f}%, Coninc:{:.2f}%, Doppler:{:.2f}%".format(
                image_name, probabs[0][0], probabs[0][1], probabs[0][2], probabs[0][3]
            )
        )


if __name__ == "__main__":
    categorization_example_use()
