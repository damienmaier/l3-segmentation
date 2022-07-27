import numpy as np


def remove_false_positive_using_threshold(image: np.ndarray, mask: np.ndarray):
    def corrected_mask_value(image_value: int, mask_value: int) -> int:
        muscle_is_possible = -29 < image_value < 150
        return mask_value if muscle_is_possible else 0

    corrected_mask = np.vectorize(corrected_mask_value)(image, mask)
    return corrected_mask
