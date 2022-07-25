import keras.models
import numpy as np

import arch.architecture
import config


def predict(images: np.ndarray, model: keras.Model, architecture: arch.architecture.Architecture):
    pre_processed_images = np.array(list(map(architecture.pre_process_image, images)))
    model_output = model.predict(pre_processed_images, batch_size=config.PREDICTION_BATCH_SIZE)
    predicted_masks = list(map(architecture.post_process_mask, model_output))

    return predicted_masks
