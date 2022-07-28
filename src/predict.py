import keras.models
import numpy as np

import config


def predict(images: np.ndarray, model: keras.Model):
    predicted_masks = model.predict(images, batch_size=config.PREDICTION_BATCH_SIZE)
    return predicted_masks
