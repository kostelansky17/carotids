from os.path import join

import matplotlib.pyplot as plt
from numpy import ndarray, uint8, zeros
from PIL import Image
from skimage.segmentation import mark_boundaries


def plot_segmentation_prediction(
    prediction: ndarray,
    label: ndarray,
    raw_img: Image,
    raw_label: Image,
    img_shape: tuple,
    img_name: str,
    save_path: str,
) -> None:
    """Trains the segmentation model on the training data.

    Parameters
    ----------
    prediction : ndarray
        The segmentatnion mask predicted by a model.
    label : ndarray
        The true segmentation mask used during the training.
    raw_img : Image
        The original image.
    raw_label : Image
        The image of the original label.
    img_shape : tuple
        The shape of the network's input. 
    img_name : str
        Name of the original input image file. 
    save_path : str
        Path to a folder to which save the figure.
    """ 
    raw_img = raw_img.resize(img_shape)

    final_mask = mark_boundaries(raw_img, prediction == 1, [255, 0, 0])
    final_mask = mark_boundaries(final_mask, prediction == 2, [0, 255, 0])
    final_mask = mark_boundaries(final_mask, prediction == 3, [0, 0, 255])

    final_seg_mask = zeros(img_shape + (3,), uint8)
    final_seg_mask[prediction == 1] = [255, 0, 0]
    final_seg_mask[prediction == 2] = [0, 255, 0] 
    final_seg_mask[prediction == 3] = [0, 0, 255]

    final_label = mark_boundaries(raw_img, label[1] & (label[0] == 0), [255, 0, 0])
    final_label = mark_boundaries(final_label, label[1] & label[0], [255, 255, 0])
    final_label = mark_boundaries(final_label, label[2], [0, 255, 0])
    
    if label.shape[0] == 4:
        final_label = mark_boundaries(final_label, label[3], [0, 0, 255])

    fig = plt.figure(figsize=(14, 14))

    fig.add_subplot(2, 2, 1)  
    plt.imshow(final_mask)
    
    plt.title("Prediction")

    fig.add_subplot(2, 2, 2)
    plt.imshow(final_seg_mask)
    plt.title("Prediction - mask")

    fig.add_subplot(2, 2, 3)
    plt.imshow(final_label)
    plt.title("Reference")

    raw_label = raw_label.resize(img_shape)
    fig.add_subplot(2, 2, 4)
    plt.imshow(raw_label)
    plt.title("Reference - mask")

    plt.savefig(join(save_path, img_name))
    plt.close()
