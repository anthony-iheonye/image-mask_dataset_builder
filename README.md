# `ImageAndMaskDatasetBuilder` Documentation

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
- [Constructor](#constructor)
- [Arguments](#arguments)
- [Output Dataset](#output-dataset)
- [Practice Examples](#practice-examples)
- [License](#license)
- [Author](#author)


## Overview

`ImageAndMaskDatasetBuilder` is a powerful utility class that streamlines the preprocessing 
of image-mask datasets for machine learning workflows. It supports a wide range of 
preprocessing steps including:

- Image and mask loading
- Cropping and resizing
- Normalization
- Multi-channel mask generation (for multi-class segmentation)
- Shuffling, batching, and prefetching

**This class constructs a modular, configurable TensorFlow input pipeline** using the `tf.data` API — making it ideal for scalable and performant training setups in image segmentation tasks.

---

## Key Features

- Automatic detection and decoding of image and mask formats (`.jpg`, `.png`, `.bmp`)
- Configurable cropping of image-mask pairs
- Optional resizing to target dimensions
- Supports grayscale, RGB, and multi-channel data
- Mask channel splitting for multi-class segmentation
- Integrated shuffling, batching, and prefetching
- Efficient input pipeline powered by TensorFlow's `tf.data.Dataset`

---
## Getting Started

### Prerequisites

- Python 3.10+
- pip ≥ 23.3.1 (Python package manager)

### a. Clone the repository

First, clone the Morph and Split repo to your local machine:

```bash
git clone https://github.com/anthony-iheonye/image-mask_dataset_builder.git
cd image-mask_dataset_builder
```

### b. Set up backend

```bash
python3 -m venv ms_venv           # create a virtual environment, ms_venv.
source ms_venv/bin/activate       # activate the ms_venv
pip install --upgrade pip         # upgrade pip
pip install -r requirements.txt   # Install dependencies
```

---

## Constructor

```python
ImageAndMaskDatasetBuilder(
    images_directory: str,
    masks_directory: str,
    image_mask_channels: Tuple[int, int],
    final_image_shape: Optional[Tuple[int, int]] = None,
    crop_image_and_mask: bool = False,
    crop_dimension: Optional[Tuple[int, int, int, int]] = None,
    normalize_image: bool = False,
    normalization_divisor: Union[int, float] = 255,
    split_mask_into_channels: bool = False,
    batch_size: Optional[int] = None,
    shuffle_buffer_size: Optional[int] = None,
    prefetch_data: bool = None,
)
```

## Arguments

| Argument                   | Type                        | Description                                                                                              |
|----------------------------|-----------------------------|----------------------------------------------------------------------------------------------------------|
| `images_directory`         | `str`                       | Directory containing the input images.                                                                   |
| `masks_directory`          | `str`                       | Directory containing the corresponding masks.                                                            |
| `image_mask_channels`      | `Tuple[int, int]`           | Tuple specifying number of channels in image and mask (e.g., `(3, 1)` for RGB image and grayscale mask). |
| `final_image_shape`        | `Tuple[int, int]`, optional | Final shape (height, width) for resizing both images and masks.                                          |
| `crop_image_and_mask`      | `bool`                      | Whether to crop image and mask before resizing.                                                          |
| `crop_dimension`           | `Tuple[int, int, int, int]` | Tuple specifying `(offset_height, offset_width, target_height, target_width)` for cropping.              |
| `normalize_image`          | `bool`                      | Whether to normalize pixel values in the image.                                                          |
| `normalization_divisor`    | `int or float`              | Divisor for normalization. `255` scales to `[0, 1]`, other values (e.g., `127.5`) scale to `[-1, 1]`.    |
| `split_mask_into_channels` | `bool`                      | Whether to split mask into multiple binary channels based on unique pixel intensities.                   |
| `batch_size`               | `int`, optional             | Batch size for the dataset. If `None`, batching is skipped.                                              |
| `shuffle_buffer_size`      | `int`, optional             | Buffer size for dataset shuffling. If `None`, no shuffling is applied.                                   |
| `prefetch_data`            | `bool`, optional            | Whether to prefetch data for performance. Uses TensorFlow AUTOTUNE.                                      |

## Output Dataset
After calling `.run()`, the `.image_mask_dataset` attribute will contain a tf.data.Dataset of
(image_tensor, mask_tensor) pairs — fully preprocessed and ready for training.

You can iterate over this dataset in training like so:

```python
builder = ImageAndMaskDatasetBuilder(...)
builder.run()
for images, masks in builder.image_mask_dataset:
    ...
```

## Practice Examples
Explore this [Jupyter Notebook](./practice_exercise.ipynb) for hands-on examples showing how to:
- Load real datasets
- Crop and resize image-mask pairs
- Normalize images
- Split masks into binary channels
- Batch, shuffle and prefetch data
- Preview data with Matplotlib


## License

This project is licensed under the MIT License.

---

## Author

Developed by [Anthony Iheonye](https://github.com/anthony-iheonye) | [LinkedIn](https://www.linkedin.com/in/anthony-iheonye/)
