"""CNN model definition used for binary glaucoma classification."""
from __future__ import annotations

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_glaucoma_cnn(
    img_height: int = 128,
    img_width: int = 128,
    dense_units: int = 128,
    dropout_rate: float = 0.25,
    learning_rate: float = 0.001,
) -> Sequential:
    """Build the compact CNN classifier used in the manuscript.

    Architecture:
    Conv2D(32, 3x3) -> MaxPool -> Conv2D(64, 3x3) -> MaxPool ->
    Conv2D(128, 3x3) -> MaxPool -> Flatten -> Dense(128) -> Dropout(0.25) ->
    Dense(64) -> Dropout(0.25) -> Dense(1, sigmoid).
    """
    model = Sequential(name="glaucoma_cnn_mcdropout")
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation="relu",
            input_shape=(img_height, img_width, 3),
            name="conv2d_32",
        )
    )
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_1"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_64"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_2"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu", name="conv2d_128"))
    model.add(MaxPooling2D(pool_size=(2, 2), name="maxpool_3"))
    model.add(Flatten(name="flatten"))
    model.add(Dense(dense_units, activation="relu", name="dense_128"))
    model.add(Dropout(dropout_rate, name="dropout_1"))
    model.add(Dense(dense_units // 2, activation="relu", name="dense_64"))
    model.add(Dropout(dropout_rate, name="dropout_2"))
    model.add(Dense(1, activation="sigmoid", name="glaucoma_probability"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
