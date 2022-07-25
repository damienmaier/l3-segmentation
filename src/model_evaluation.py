import random
import statistics

import sklearn
from matplotlib import pyplot as plt

import utils.display_image
import pandas
import seaborn


def dice_coefficient(mask1, mask2):
    return sklearn.metrics.f1_score(mask1.ravel(), mask2.ravel())


def dice_coefficients(masks1, masks2):
    return list(map(dice_coefficient, masks1, masks2))


def average_dice_coefficient(masks1, masks2):
    dice_coefficients_list = dice_coefficients(masks1, masks2)
    return statistics.mean(dice_coefficients_list)


def model_performance_summary(images, true_masks, predicted_masks):
    for image, true_mask, predicted_mask in random.sample(list(zip(images, true_masks, predicted_masks)), 20):
        utils.display_image.display_ct_scan_image_and_two_masks(image=image, blue_mask=true_mask,
                                                                red_mask=predicted_mask)

    dice_coefficients_list = dice_coefficients(true_masks, predicted_masks)
    seaborn.catplot(
        data=(pandas.DataFrame({"dice coefficient": dice_coefficients_list})),
        y="dice coefficient",
        kind="box")
    plt.show()

    print(f"Average dice coefficient : {statistics.mean(dice_coefficients_list):.3f}")
