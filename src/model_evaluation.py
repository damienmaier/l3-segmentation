import random
import statistics

import pandas
import seaborn
import skimage.metrics
import tensorflow as tf
from matplotlib import pyplot as plt

import utils.display_image


def dice_coefficient_between_two_masks_tensor(mask1: tf.Tensor, mask2: tf.Tensor) -> tf.Tensor:
    mask1_bool = mask1 == 1
    mask2_bool = mask2 == 1

    intersection_mask = tf.math.logical_and(mask1_bool, mask2_bool)
    intersection_size = tf.math.count_nonzero(intersection_mask)

    mask1_size = tf.math.count_nonzero(mask1)
    mask2_size = tf.math.count_nonzero(mask2)

    return 2 * intersection_size / (mask1_size + mask2_size)


def dice_coefficient_between_two_masks(mask1, mask2) -> float:
    return float(dice_coefficient_between_two_masks_tensor(mask1, mask2))


def dice_coefficients_between_mask_batches(mask_batch1: tf.Tensor, mask_batch2: tf.Tensor):
    def dice_coefficient_between_two_stacked_masks(tensor):
        return dice_coefficient_between_two_masks_tensor(tensor[0], tensor[1])

    stacked_masks = tf.stack([mask_batch1, mask_batch2], axis=1)
    dice_coefficients = tf.vectorized_map(dice_coefficient_between_two_stacked_masks, stacked_masks)
    return dice_coefficients


hausdorff_distance_between_two_masks = skimage.metrics.hausdorff_distance


def model_performance_summary(images, blue_masks, red_masks, images_display_count: int, blue_mask_legend,
                              red_mask_legend):
    for image, blue_mask, red_mask in random.sample(list(zip(images, blue_masks, red_masks)),
                                                    images_display_count):
        utils.display_image.display_ct_scan_image(image)
        utils.display_image.display_ct_scan_image_and_mask(image, red_mask)
        utils.display_image.display_ct_scan_image_and_two_masks(image=image,
                                                                blue_mask=blue_mask, red_mask=red_mask,
                                                                blue_mask_legend=blue_mask_legend,
                                                                red_mask_legend=red_mask_legend)

    dice_values = list(map(dice_coefficient_between_two_masks, blue_masks, red_masks))
    hausdorff_values = list(map(hausdorff_distance_between_two_masks, blue_masks, red_masks))

    _display_metric_box_plot(dice_values, "dice coefficient")
    _display_metric_box_plot(hausdorff_values, "hausdorff distance")

    print(f"Average dice coefficient : {statistics.mean(dice_values):.4f}")
    print(f"Average hausdorff distance : {statistics.mean(hausdorff_values):.4f}")


def _display_metric_box_plot(metric_values, metric_name):
    seaborn.catplot(
        data=(pandas.DataFrame({metric_name: metric_values})),
        y=metric_name,
        kind="box")
    plt.show()
