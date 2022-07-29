import random

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


def dice_coefficients_between_multiple_pairs_of_masks(masks1: tf.Tensor, masks2: tf.Tensor):
    def dice_coefficient_between_two_stacked_masks(tensor):
        return dice_coefficient_between_two_masks(tensor[0], tensor[1])

    stacked_masks = tf.stack([masks1, masks2], axis=1)
    dice_coefficients = tf.vectorized_map(dice_coefficient_between_two_stacked_masks, stacked_masks)
    return dice_coefficients


def average_dice_coefficient(masks1: tf.Tensor, masks2: tf.Tensor):
    dice_coefficients = dice_coefficients_between_multiple_pairs_of_masks(masks1, masks2)
    return tf.reduce_mean(dice_coefficients)


def model_performance_summary(images, true_masks, predicted_masks):
    for image, true_mask, predicted_mask in random.sample(list(zip(images, true_masks, predicted_masks)), 2):
        utils.display_image.display_ct_scan_image(image)
        utils.display_image.display_ct_scan_image_and_mask(image, predicted_mask)
        utils.display_image.display_ct_scan_image_and_two_masks(image=image, blue_mask=true_mask,
                                                                red_mask=predicted_mask)

    dice_coefficients_list = dice_coefficients_between_multiple_pairs_of_masks(true_masks, predicted_masks)
    seaborn.catplot(
        data=(pandas.DataFrame({"dice coefficient": dice_coefficients_list})),
        y="dice coefficient",
        kind="box")
    plt.show()

    print(f"Average dice coefficient : {average_dice_coefficient(true_masks, predicted_masks):.4f}")
