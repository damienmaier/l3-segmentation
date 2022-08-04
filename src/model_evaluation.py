import functools
import itertools
import random
import statistics

import numpy as np
import pandas
import seaborn
import skimage.draw
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


def hausdorff_points(mask1, mask2):
    point1, point2 = skimage.metrics.hausdorff_pair(mask1, mask2)
    return list(point1), list(point2)


def model_performance_summary(images,
                              blue_masks, red_masks,
                              blue_mask_legend, red_mask_legend,
                              images_display_count: int, selection=None,
                              display_image=False, display_red_mask=False, display_blue_mask=False,
                              display_comparison=True, display_comparison_with_hausdorff_line=False,
                              display_box_plots=False,
                              alternative_red_masks=None):
    if alternative_red_masks is not None:
        images_masks = list(zip(images, blue_masks, red_masks, alternative_red_masks))
    else:
        images_masks = list(zip(images, blue_masks, red_masks, itertools.repeat(None)))
    if selection == "random":
        random.shuffle(images_masks)
    if selection == "hausdorff":
        images_masks.sort(
            key=lambda image_mask: hausdorff_distance_between_two_masks(image_mask[1], image_mask[2]),
            reverse=True
        )
    if selection == "dice":
        images_masks.sort(
            key=lambda image_mask: dice_coefficient_between_two_masks(image_mask[1], image_mask[2]),
        )

    for image, blue_mask, red_mask, alternative_red_mask in images_masks[:images_display_count]:
        if display_image:
            utils.display_image.display_ct_scan_image(image)
        if display_red_mask:
            utils.display_image.display_ct_scan_image_and_mask(image, red_mask)
        if display_blue_mask:
            utils.display_image.display_ct_scan_image_and_mask(image, blue_mask)

        display_ct_scan_image_and_two_masks = functools.partial(utils.display_image.display_ct_scan_image_and_two_masks,
                                                                image=image,
                                                                blue_mask=blue_mask,
                                                                blue_mask_legend=blue_mask_legend,
                                                                red_mask_legend=red_mask_legend
                                                                )

        if display_comparison:
            display_ct_scan_image_and_two_masks(red_mask=red_mask, show_hausdorff_location=False)
            if alternative_red_masks is not None:
                display_ct_scan_image_and_two_masks(red_mask=alternative_red_mask, show_hausdorff_location=False)

        if display_comparison_with_hausdorff_line:
            display_ct_scan_image_and_two_masks(red_mask=red_mask, show_hausdorff_location=True)
            if alternative_red_masks is not None:
                display_ct_scan_image_and_two_masks(red_mask=alternative_red_mask, show_hausdorff_location=True)

    dice_values = list(map(dice_coefficient_between_two_masks, blue_masks, red_masks))
    hausdorff_values = list(map(hausdorff_distance_between_two_masks, blue_masks, red_masks))
    if display_box_plots:
        utils.display_image.display_metric_box_plot(dice_values, "dice coefficient")
        utils.display_image.display_metric_box_plot(hausdorff_values, "hausdorff distance")

    print(f"Average dice coefficient : {statistics.mean(dice_values):.4f}")
    print(f"Average hausdorff distance : {statistics.mean(hausdorff_values):.4f}")



