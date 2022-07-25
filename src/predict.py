import keras.models
import numpy as np

import config
import dataset.data_loading
import rootdir

from architectures.joachim import pre_process, post_process
from model_training import MODEL_PATH

TEST_SET_PREDICTIONS_PATH = rootdir.PROJECT_ROOT_PATH / "test set predicted masks.npy"

model = keras.models.load_model(MODEL_PATH)


def predict(images):
    pre_processed_images = np.array(list(map(pre_process, images)))
    model_output = model.predict(pre_processed_images, batch_size=config.PREDICTION_BATCH_SIZE)
    predicted_masks = list(map(post_process, model_output))

    return predicted_masks


