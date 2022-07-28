import keras
import numpy as np
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


def _train_model(images: np.ndarray, masks: np.ndarray) -> keras.Model:
    images_dataset = tf.data.Dataset.from_tensor_slices(images)
    masks_dataset = tf.data.Dataset.from_tensor_slices(masks)
    dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

    batched_dataset = dataset.shuffle(len(images)).batch(config.TRAINING_BATCH_SIZE).prefetch(buffer_size=1)

    model = arch.model_building.joachim.model_sma_detection()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=model_evaluation.dice_metric_for_tf_model()
    )

    model.fit(
        x=batched_dataset,
        epochs=10
    )

    return model
