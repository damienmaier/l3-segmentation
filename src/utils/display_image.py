"""
Functions for generating images of CT scans and masks.

If config.SAVE_IMAGES_ON_DISK is False, the images are displayed in a window.
If it is True, the images are saved in the directory <project root>/images.
"""

import pathlib

import matplotlib
import numpy as np
import pandas
import seaborn
import skimage.metrics
from matplotlib import pyplot as plt

import config
import model_evaluation
import rootdir

IMAGES_FOLDER_PATH = pathlib.Path(rootdir.PROJECT_ROOT_PATH / "images")

MASK_COLOR_MAP = matplotlib.colormaps["jet_r"]


def display_ct_scan_image(image: np.ndarray):
    """
    Generates an image of a CT scan image by clipping pixel HU values to [-200, 200]
    """
    _plot_ct_scan_image(image)
    _finalize_image()


def display_mask(mask: np.ndarray):
    """
    Generates an image of a mask
    """
    _plot_image(mask)
    _finalize_image()


def display_ct_scan_image_and_mask(image: np.ndarray, mask: np.ndarray):
    """
    Generates an image of a CT scan image where an area is highlighted with respect to a mask.
    """
    _plot_ct_scan_image(image)
    _plot_mask(mask)
    _finalize_image()


def display_ct_scan_image_and_two_masks(image: np.ndarray, blue_mask: np.ndarray, red_mask: np.ndarray,
                                        blue_mask_legend: str, red_mask_legend: str,
                                        show_hausdorff_location: bool = True):
    """
    Generates an image that allows to visualize the differences between two masks for a CT scan image.

    Images areas that are in both masks are highlighted in green. Areas that are only in `blue_mask` are highlighted in
    blue and areas that are only in `red_mask` are highlighted in red.

    The image includes a legend that names the masks with respect to `blue_mask_legend` and `red_mask_legend`.

    The image also includes the Dice coefficient and the Hausdorff distance between the two masks.

    If `show_hausdorff_location` is True, the image includes a line that shows the location of the Hausdorff distance.
    """
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
        matplotlib.lines.Line2D([0], [0], color=MASK_COLOR_MAP(0.), label=f"{red_mask_legend} only"),
        matplotlib.lines.Line2D([0], [0], color=MASK_COLOR_MAP(0.5),
                                label=f"both {red_mask_legend} and {blue_mask_legend}"),
        matplotlib.lines.Line2D([0], [0], color=MASK_COLOR_MAP(1.), label=f"{blue_mask_legend} only"),
    ]
    plt.legend(handles=legend_elements, loc="upper left", fontsize=15)

    dice_coefficient = model_evaluation.dice_coefficient_between_two_masks(blue_mask, red_mask)
    hausdorff_distance = model_evaluation.hausdorff_distance_between_two_masks(blue_mask, red_mask)
    plt.text(x=20, y=490,
             s=f"Dice coefficient : {dice_coefficient :.2f}\nHausdorff distance : {hausdorff_distance :.1f}",
             backgroundcolor="white",
             fontsize=15)
    if show_hausdorff_location:
        hausdorff_point_1, hausdorff_point_2 = skimage.metrics.hausdorff_pair(blue_mask, red_mask)
        plt.plot([hausdorff_point_1[1], hausdorff_point_2[1]], [hausdorff_point_1[0], hausdorff_point_2[0]],
                 alpha=1, color="m", linewidth=1)
    _finalize_image()


def _plot_image(image, *args, **kwargs):
    y_shape, x_shape = image.shape
    scale_factor = 2
    plt.rcParams["figure.figsize"] = (x_shape / 100 * scale_factor, y_shape / 100 * scale_factor)
    plt.subplots_adjust(0, 0, 1, 1)
    plt.imshow(image, interpolation="none", *args, **kwargs)


def _plot_ct_scan_image(image):
    _plot_image(image, cmap="Greys_r", vmin=-200, vmax=200)


def _plot_mask(mask):
    mask_as_masked_np_array = np.ma.masked_equal(mask, 0)
    _plot_image(mask_as_masked_np_array, cmap=MASK_COLOR_MAP, alpha=.3,
                norm=matplotlib.colors.Normalize(vmin=1, vmax=3))


def _finalize_image(display_axis=False):
    if not display_axis:
        plt.axis("off")
    if not config.SAVE_IMAGES_ON_DISK:
        plt.show()
    else:
        if not IMAGES_FOLDER_PATH.exists():
            IMAGES_FOLDER_PATH.mkdir()

        last_image_stem = max((file.stem for file in IMAGES_FOLDER_PATH.iterdir()), default=0)
        image_name = str(int(last_image_stem) + 1).zfill(4) + ".png"
        plt.savefig(IMAGES_FOLDER_PATH / image_name)
        plt.clf()


def display_metric_box_plot(metric_values: list[float], metric_name: str):
    """
    Generates an image of a box plot for the values listed in `metric_values`. The image includes a legend
    that uses `metric_name`.
    """
    seaborn.catplot(
        data=(pandas.DataFrame({metric_name: metric_values})),
        y=metric_name,
        kind="box")
    _finalize_image(display_axis=True)
