"""Image preprocessing and Keras data generators for fundus images."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from skimage import color, exposure, filters, io, transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_fundus_images(
    image_paths: Iterable[str | Path],
    target_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Preprocess fundus images exactly as used in the manuscript pipeline.

    Steps:
    1. read RGB image;
    2. convert to grayscale;
    3. resize to 128 x 128;
    4. histogram equalisation;
    5. Gaussian filtering with sigma = 1;
    6. stack the single channel into three channels for the CNN input.

    Parameters
    ----------
    image_paths:
        List or iterable of image paths.
    target_size:
        Output spatial size, default (128, 128).

    Returns
    -------
    np.ndarray
        Array with shape (N, H, W, 3), dtype float32.
    """
    images = []
    for path in image_paths:
        image = io.imread(str(path))
        image_gray = color.rgb2gray(image)
        image_resized = transform.resize(
            image_gray,
            target_size,
            anti_aliasing=True,
            preserve_range=False,
        )
        image_equalized = exposure.equalize_hist(image_resized)
        image_filtered = filters.gaussian(image_equalized, sigma=1)
        image_rgb = np.stack((image_filtered,) * 3, axis=-1)
        images.append(image_rgb.astype(np.float32))
    return np.asarray(images, dtype=np.float32)


def make_generators(
    data_dir: str | Path,
    img_size: Tuple[int, int] = (128, 128),
    batch_size: int = 28,
    validation_split: float = 0.30,
    shuffle_train: bool = False,
    shuffle_validation: bool = False,
):
    """Create training and validation generators.

    The augmentation matches the original manuscript code: rescale, shear, zoom,
    horizontal flip, and 70/30 split.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    train_generator = datagen.flow_from_directory(
        str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=shuffle_train,
    )

    validation_generator = datagen.flow_from_directory(
        str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=shuffle_validation,
    )

    return train_generator, validation_generator
