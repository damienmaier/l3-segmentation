import keras
import keras_tuner
import numpy as np
import sklearn.model_selection
import tensorflow as tf

import arch.custom_layers
import arch.model_building.joachim
import config
import model_evaluation


def get_best_model(images: np.ndarray, masks: np.ndarray) -> keras.Model:
    best_model = _train_model(images, masks)
    x = best_model.output
    x = arch.custom_layers.RoundLayer()(x)
    best_model_with_final_round_layer = keras.Model(best_model.input, x)
    return best_model_with_final_round_layer


def _create_dataset_from_data(images: np.ndarray, masks: np.ndarray, batch_size: int) -> tf.data.Dataset:
    base_dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    prepared_dataset = base_dataset.shuffle(len(images)).batch(batch_size).prefetch(buffer_size=1)
    return prepared_dataset


def _train_model(images: np.ndarray, masks: np.ndarray) -> keras.Model:
    images_train, images_validation, masks_train, masks_validation = \
        sklearn.model_selection.train_test_split(images, masks)

    train_dataset = _create_dataset_from_data(images_train, masks_train, config.TRAINING_BATCH_SIZE)
    validation_dataset = _create_dataset_from_data(images_validation, masks_validation, config.PREDICTION_BATCH_SIZE)

    model = arch.model_building.joachim.model_sma_detection()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=model_evaluation.dice_metric_for_tf_model
    )

    model.fit(
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=100
    )

    return model

