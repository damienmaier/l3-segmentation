import keras
import numpy as np
import tensorflow as tf
import arch.architecture
import arch.model_building.joachim

BATCH_SIZE = 100


def get_best_model(images: np.ndarray, masks: np.ndarray) -> keras.Model:
    return _train_model(images, masks, arch.architecture.joachim)


def _train_model(images: np.ndarray, masks: np.ndarray, architecture: arch.architecture.Architecture) -> keras.Model:
    preprocessed_images = np.array(list(map(architecture.pre_process_image, images)))
    images_dataset = tf.data.Dataset.from_tensor_slices(preprocessed_images)
    masks_dataset = tf.data.Dataset.from_tensor_slices(masks)
    dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

    batched_dataset = dataset.shuffle(len(images)).batch(BATCH_SIZE).prefetch(buffer_size=1)

    model = architecture.build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
    )

    model.fit(
        x=batched_dataset,
        epochs=100
    )

    return model
