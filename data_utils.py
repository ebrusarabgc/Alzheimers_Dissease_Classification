import numpy as np
import tensorflow as tf
import pathlib


def load_data(data_dir, val_split, image_size, batch_size, color_mode, seed):
    path_dir = pathlib.Path(data_dir)
    class_names = np.array(sorted([item.name for item in path_dir.glob('*')]))

    tf.random.set_seed(seed)
    np.random.seed(seed)

    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=val_split,
        subset="training",
        seed=seed,
        color_mode=color_mode,
        image_size=image_size,
        batch_size=batch_size,
    )

    valid_data = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        label_mode='categorical',
        seed=seed,
        color_mode=color_mode,
        image_size=image_size,
        batch_size=batch_size,
    )

    return train_data, valid_data, class_names
