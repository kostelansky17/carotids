import os

import numpy as np
from PIL import Image


COLOR_MAX = 255
crop_shape = (181, 141, 831, 750)

PATH_LONG = "/home/martin/Documents/cartroids/data/categorization/long"
PATH_TRAV = "/home/martin/Documents/cartroids/data/categorization/trav"
PATH_DIFF = "/home/martin/Documents/cartroids/data/categorization/diff"

CROP_BIG = (221, 171, 791, 740)
CROP_SMALL = (181, 141, 831, 750)
SIZE_BIG = (1200, 900)
SIZE_SMALL = (1200, 870)


def load_dir(dir_path, crop=False):
    data = []
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        img = convert_img_to_feature(file_path, crop_shape)
        data.append(img)

    return np.vstack(data)


def crop_image(img):
    if img.size == SIZE_BIG:
        img = img.crop(CROP_BIG)
    elif img.size == SIZE_SMALL:
        img = img.crop(CROP_SMALL)

    return img


def convert_img_to_feature(img_path, crop=False):
    img = Image.open(img_path).convert("LA")
    if crop:
        img = crop_image(img)

    img = img.resize((50, 50), Image.ANTIALIAS)
    return np.asarray(img).flatten() / COLOR_MAX


def create_categorization_features():
    X = np.vstack((load_dir(PATH_LONG), load_dir(PATH_TRAV), load_dir(PATH_DIFF, True)))
    y = np.concatenate(
        (
            np.ones(len(os.listdir(PATH_LONG))),
            np.zeros(len(os.listdir(PATH_TRAV))),
            np.ones(len(os.listdir(PATH_DIFF)))*-1,
        )
    )

    return X, y
