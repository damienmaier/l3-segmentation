import dataclasses
from typing import Callable

import keras
import numpy as np

import preprocessing
import arch.model_building.joachim


@dataclasses.dataclass
class Architecture:
    build_model: Callable[[], keras.Model]
    """Function that returns a new freshly built model for this arch"""

    pre_process_image: Callable[[np.ndarray], np.ndarray]
    """Function that takes an image as argument and transforms it adequately to be given as an input to the model

    The image argument is a 2D numpy array containing HU values.
    The output is a numpy array.
    """

    post_process_mask: Callable[[np.ndarray], np.ndarray]
    """Function that takes as argument a row output of the model and transforms it to an actual mask prediction

    The output is a 2D numpy array where each element is either 0 (no part of the mask) or 1 (is part of the mask)
    """


def _joachim_preprocessing(image: np.ndarray):
    image = preprocessing.clip(image)
    image = preprocessing.single_channel(image)
    return image


def _joachim_postprocessing(model_output: np.ndarray):
    model_output_2d = np.squeeze(model_output)
    predicted_mask = (model_output_2d > 0.5).astype(int)
    return predicted_mask


joachim = Architecture(
    build_model=arch.model_building.joachim.model_sma_detection,
    pre_process_image=_joachim_preprocessing,
    post_process_mask=_joachim_postprocessing
)
