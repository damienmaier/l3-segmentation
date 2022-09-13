"""
Functions for measuring the performance of the model by comparing predicted masks with correct masks.
"""

import functools
import itertools
import random
import statistics

import skimage.draw
import skimage.metrics
import tensorflow as tf

import utils.display_image


def tensor_dice_coefficient_between_two_masks(mask1, mask2) -> tf.Tensor:
    """
    Computes the dice coefficient between `mask1` and `mask2`.

    `mask1` and `mask2` are 2D arrays of same dimension that contain only 0s and 1s. They can be tf tensors or np arrays.

    Returns the dice coefficient as a scalar tf tensor.
    """
    mask1_bool = mask1 == 1
    mask2_bool = mask2 == 1

    intersection_mask = tf.math.logical_and(mask1_bool, mask2_bool)
    intersection_size = tf.math.count_nonzero(intersection_mask)

    mask1_size = tf.math.count_nonzero(mask1)
    mask2_size = tf.math.count_nonzero(mask2)

    return 2 * intersection_size / (mask1_size + mask2_size)


def dice_coefficient_between_two_masks(mask1, mask2) -> float:
    """
    Computes the dice coefficient between `mask1` and `mask2`.

    `mask1` and `mask2` are 2D arrays of same dimension that contain only 0s and 1s. They can be tf tensors or np arrays.

    Returns the dice coefficient as a float.
    """
    return float(tensor_dice_coefficient_between_two_masks(mask1, mask2))


def dice_coefficients_between_mask_batches(mask_batch1: tf.Tensor, mask_batch2: tf.Tensor) -> tf.Tensor:
    """
    Returns the dice coefficients between two mask batches.

    `mask_batch1` and `mask_batch2` both have shape (<batch size>, <dim1>, <dim2>).

    Computes the dice coefficients between the fist masks of each batch, then between the second masks of each batch, etc.

    Returns a 1D tf tensor of shape (<batch size>) containing the computed dice coefficients.
    """
    def dice_coefficient_between_two_stacked_masks(tensor):
        return tensor_dice_coefficient_between_two_masks(tensor[0], tensor[1])

    stacked_masks = tf.stack([mask_batch1, mask_batch2], axis=1)
    dice_coefficients = tf.vectorized_map(dice_coefficient_between_two_stacked_masks, stacked_masks)
    return dice_coefficients


hausdorff_distance_between_two_masks = skimage.metrics.hausdorff_distance


def hausdorff_points(mask1, mask2) -> tuple[list[int, int], list[int, int]]:
    """
    Computes the coordinates of the points for the Hausdorff distance between two masks.
    """
    point1, point2 = skimage.metrics.hausdorff_pair(mask1, mask2)
    return list(point1), list(point2)


def model_performance_summary(images,
                              blue_masks, red_masks,
                              blue_mask_legend: str, red_mask_legend: str,
                              images_display_count: int, selection=None,
                              display_image=False, display_red_mask=False, display_blue_mask=False,
                              display_comparison=True, display_comparison_with_hausdorff_line=False,
                              display_box_plots=False,
                              alternative_red_masks=None):
    """
    Displays several images to measure and visualize the differences between two set of masks
    for a set of input images to be segmented. Prints the average dice coefficient and Hausdorff distance between the
    two sets of masks.

    `images` is an iterable where each element is a 2D array corresponding to an image.
    `blue_masks` and `red_masks` are iterables where each element is a 2D array corresponding to a mask.

    The function will choose `images_display_count` images to display.

    `selection` can be
        - None : the first `images_display_count` are used
        - "random" : the images are chosen randomly
        - "hausdorff" : the chosen images are the one where the Hausdorff distance between the red mask and the blue mask is the worst
        - "dice" : the chosen images are the one where the dice coefficient between the red mask and the blue mask is the worst

    For each chosen image :
        - `display_image` displays the image alone, without the masks
        -  `display_red_mask` displays the image with the red mask area highlighted in red
        -  `display_blue_mask` displays the image with the blue mask area highlighted in red
        -  `display_comparison` displays the image with both masks in a way the allows to compare them easily and with the dice coefficient and Hausdorff distance
        -  `display_comparison_with_hausdorff_line` does the same, but also the line for the Hausdorff distance is displayed

    `blue_mask_legend` and `red_mask_legend` are string that describe the blue masks and the red masks respectively.
    They are used for generating the legend when `display_comparison` or `display_comparison_with_hausdorff_line`
    is True.

    `alternative_red_masks` is an iterable where each element is a 2D array corresponding to a mask. If provided
    and if `display_comparison` or `display_comparison_with_hausdorff_line` is true, a second image will be generated
    each time with the mask from `alternative_red_masks` instead of the one from `red_masks`

    If `display_box_plots` is True, box plots for the dice coefficients and for the Hausdorff values are displayed.
    Those values are computed on the whole provided data, not only on the data that is displayed.
    """
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
