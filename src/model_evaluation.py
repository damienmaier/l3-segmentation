import random
import statistics

import pandas
import seaborn
import tensorflow as tf
from matplotlib import pyplot as plt

import utils.display_image


def dice_coefficient_between_two_masks(mask1: tf.Tensor, mask2: tf.Tensor):
    mask1_bool = mask1 == 1
    mask2_bool = mask2 == 1

    intersection_mask = tf.math.logical_and(mask1_bool, mask2_bool)
    intersection_size = tf.math.count_nonzero(intersection_mask)

    mask1_size = tf.math.count_nonzero(mask1)
    mask2_size = tf.math.count_nonzero(mask2)

    return 2 * intersection_size / (mask1_size + mask2_size)


def dice_coefficients_between_mask_batches(mask_batch1: tf.Tensor, mask_batch2: tf.Tensor):
    def dice_coefficient_between_two_stacked_masks(tensor):
        return dice_coefficient_between_two_masks(tensor[0], tensor[1])

    stacked_masks = tf.stack([mask_batch1, mask_batch2], axis=1)
    dice_coefficients = tf.vectorized_map(dice_coefficient_between_two_stacked_masks, stacked_masks)
    return dice_coefficients


def dice_coefficients_between_masks_iterables(masks1, masks2):
    coefficients_tensor = map(dice_coefficient_between_two_masks, masks1, masks2)
    coefficients_float = list(map(float, coefficients_tensor))
    return coefficients_float


def model_performance_summary(images, true_masks, predicted_masks):
    for image, true_mask, predicted_mask in random.sample(list(zip(images, true_masks, predicted_masks)), 2):
        utils.display_image.display_ct_scan_image(image)
        utils.display_image.display_ct_scan_image_and_mask(image, predicted_mask)
        utils.display_image.display_ct_scan_image_and_two_masks(image=image, blue_mask=true_mask,
                                                                red_mask=predicted_mask)

    dice_coefficients_list = dice_coefficients_between_masks_iterables(true_masks, predicted_masks)
    seaborn.catplot(
        data=(pandas.DataFrame({"dice coefficient": dice_coefficients_list})),
        y="dice coefficient",
        kind="box")
    plt.show()

    print(f"Average dice coefficient : {statistics.mean(dice_coefficients_list):.4f}")
