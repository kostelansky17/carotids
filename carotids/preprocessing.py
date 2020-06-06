import os

import numpy as np
from PIL import Image


COLOR_MAX = 255
crop_shape = (181, 141, 831, 750)

PATH_LONG = "/home/martin/Documents/cartroids/data/categorization/test/praha_long"
PATH_TRAV = "/home/martin/Documents/cartroids/data/categorization/test/praha_trav"
PATH_DIFF = "/home/martin/Documents/cartroids/data/categorization/test/praha_diff"

CROP_BIG = (221, 171, 791, 740)
CROP_SMALL = (181, 141, 831, 750)
SIZE_BIG = (1200, 900)
SIZE_SMALL = (1200, 870)


def load_dir(dir_path, crop=False):
    data = []
    for file in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file)
        img = convert_img_to_feature(file_path, crop)
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

    img = img.resize((50, 50))
    return np.asarray(img)[:, :, 0].flatten() / COLOR_MAX


def load_position(dir_path, label_file):
    label = np.loadtxt(os.path.join(dir_path, label_file), delimiter=";")

    transformed_label = np.asarray(
        [
            label[0] - label[2] / 2,
            label[1] - label[3] / 2,
            label[0] + label[2] / 2,
            label[1] + label[3] / 2,
        ]
    )

    return transformed_label


def load_imgs_dir(dir_path):
    data = []
    for file in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file)
        img = Image.open(file_path)
        data.append(img)

    return data


def load_img(dir_path, img_file):
    joined_path = os.path.join(dir_path, img_file)
    img = Image.open(joined_path)

    return img


def normalize_data(X_train, X_test=None):
    train_mean = np.mean(X_train)
    train_std = np.mean(X_train)

    X_train = (X_train - train_mean) / train_std

    if X_test:
        X_test = (X_test - train_mean) / train_std

        return X_train, X_test

    return X_train
