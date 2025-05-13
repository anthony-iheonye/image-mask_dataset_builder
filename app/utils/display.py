import tensorflow as tf
from typing import Union

import matplotlib.pyplot as plt
import tensorflow as tf


def imshow(image, title=None):
    """Displays an image with corresponding title."""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)

def show_images_with_title(images: Union[list, tuple], titles: Union[list, tuple]):
    """Displays a row of images (along with their titles)"""
    if len(images) != len(titles):
        raise ValueError(f"titles are not complete, got {titles}")

    plt.figure(figsize=(20, 12))
    for idx, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), idx + 1)
        plt.xticks([])
        plt.yticks([])
        imshow(image, title)


