import functools

import matplotlib
import numpy as np
import skimage.metrics
from matplotlib import pyplot as plt

import model_evaluation

MASK_COLOR_MAP = matplotlib.colormaps["jet_r"]


def display_ct_scan_image(image):
    _plot_ct_scan_image(image)
    plt.axis("off")
    plt.show()


def display_mask(mask):
    _plot_image(mask)
    plt.axis("off")
    plt.show()


def display_ct_scan_image_and_mask(image, mask):
    _plot_ct_scan_image(image)
    _plot_mask(mask)
    plt.axis("off")
    plt.show()


def display_ct_scan_image_and_two_masks(image, blue_mask, red_mask, blue_mask_legend, red_mask_legend,
                                        show_hausdorff_location: bool = True):
    def comparison_mask_value(blue_mask_value, red_mask_value):
        if blue_mask_value == 0 and red_mask_value == 0:
            return 0
        if blue_mask_value == 0 and red_mask_value == 1:
            return 1
        if blue_mask_value == 1 and red_mask_value == 1:
            return 2
        if blue_mask_value == 1 and red_mask_value == 0:
            return 3

    comparison_mask = np.vectorize(comparison_mask_value)(blue_mask, red_mask)

    _plot_ct_scan_image(image)
    _plot_mask(comparison_mask)

    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color=MASK_COLOR_MAP(0.), label=red_mask_legend),
        matplotlib.lines.Line2D([0], [0], color=MASK_COLOR_MAP(1.), label=blue_mask_legend)
    ]
    plt.legend(handles=legend_elements, loc="upper left")

    dice_coefficient = model_evaluation.dice_coefficient_between_two_masks(blue_mask, red_mask)
    hausdorff_distance = model_evaluation.hausdorff_distance_between_two_masks(blue_mask, red_mask)
    plt.text(x=20, y=490,
             s=f"Dice coefficient : {dice_coefficient :.2f}\nHausdorff distance : {hausdorff_distance :.1f}",
             backgroundcolor="white")
    if show_hausdorff_location:
        hausdorff_point_1, hausdorff_point_2 = skimage.metrics.hausdorff_pair(blue_mask, red_mask)
        plt.plot([hausdorff_point_1[1], hausdorff_point_2[1]], [hausdorff_point_1[0], hausdorff_point_2[0]],
                 alpha=1, color="m", linewidth=1)
    plt.axis("off")
    plt.show()


_plot_image = functools.partial(plt.imshow, interpolation="none")


def _plot_ct_scan_image(image):
    _plot_image(image, cmap="Greys_r", vmin=-200, vmax=200)


def _plot_mask(mask):
    mask_as_masked_np_array = np.ma.masked_equal(mask, 0)
    _plot_image(mask_as_masked_np_array, cmap=MASK_COLOR_MAP, alpha=.3,
                norm=matplotlib.colors.Normalize(vmin=1, vmax=3))
