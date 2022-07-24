import keras.models
import numpy as np
import tensorflow as tf

import utils.preprocessing
from architectures.keras_example import DeeplabV3Plus
from dataset.data_loading import get_train_set
from rootdir import PROJECT_ROOT_PATH

from utils.preprocessing import triple_channels, single_channel
import architectures.joachim

MODEL_PATH = PROJECT_ROOT_PATH / "model"

BATCH_SIZE = 10


def get_best_model():
    images, masks = get_train_set()
    _train_model(images, masks)


def _train_model(X, Y):
    X = np.array(list(map(utils.preprocessing.clip, X)))
    X = np.array(list(map(single_channel, X)))
    X_dataset = tf.data.Dataset.from_tensor_slices(X)
    Y_dataset = tf.data.Dataset.from_tensor_slices(Y)
    dataset = tf.data.Dataset.zip((X_dataset, Y_dataset))

    batched_dataset = dataset.shuffle(len(X)).batch(BATCH_SIZE).prefetch(buffer_size=1)

    model = architectures.joachim.model_sma_detection((512, 512, 1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
    )

    # model = keras.models.load_model(MODEL_PATH)

    model.fit(
        x=batched_dataset,
        epochs=100
    )

    model.save(MODEL_PATH)
