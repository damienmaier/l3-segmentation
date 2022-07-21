import numpy as np
import tensorflow as tf

from architectures.keras_example import DeeplabV3Plus
from rootdir import PROJECT_ROOT_PATH
from utils.preprocessing import triple_channels, single_channel
import architectures.joachim

BATCH_SIZE = 10


def train_best_model(X, Y):
    _train_model(X, Y)


def _train_model(X, Y):
    X = np.array(list(map(single_channel, X)))
    X_dataset = tf.data.Dataset.from_tensor_slices(X)
    Y_dataset = tf.data.Dataset.from_tensor_slices(Y)
    dataset = tf.data.Dataset.zip((X_dataset, Y_dataset))

    batched_dataset = dataset.shuffle(len(X)).batch(BATCH_SIZE).prefetch(buffer_size=1)

    model = architectures.joachim.model_sma_detection((512, 512, 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
        loss="binary_crossentropy",
    )

    model.fit(
        x=batched_dataset,
        epochs=5
    )

    model.save(PROJECT_ROOT_PATH / "model")
