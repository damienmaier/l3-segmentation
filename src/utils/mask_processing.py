"""
Functions for processing masks.

A mask is a 2D numpy array containing only 0s and 1s.
"""

import collections
import math

import numpy as np
import scipy.spatial
import skimage.measure
import skimage.morphology


def remove_isolated_areas(mask: np.ndarray, min_ratio: float, max_distance: int) -> np.ndarray:
    """
    Remove isolated areas that are too far away from the main areas of the muscle.

    The largest isolated area and all areas whose size is at least `min_ratio` of the size of the largest area are
    considered to be part of the main area and are not removed.

    Isolated areas whose distance to the main area is less than `max_distance` are also considered part of main area.

    Remaining isolated areas are removed.

    Returns the resulting mask. The original mask is not modified.
    """
    labeled_mask, objects_count = skimage.measure.label(mask, return_num=True)

    if objects_count == 0:
        return mask

    pixels_counter = collections.Counter(labeled_mask.flat)
    largest_objects = pixels_counter.most_common()
    # the most common value is 0, so the id of the largest object is the second most common value
    largest_object_id, largest_object_size = largest_objects[1]
    other_objects = largest_objects[2:]

    # to include an object in the final mask, we give it the value largest_object_id

    # find the main mask areas
    # the objects whose size is close enough to the size of the largest object are part of the main mask area
    for object_id, object_size in other_objects:
        if object_size >= min_ratio * largest_object_size:
            labeled_mask[labeled_mask == object_id] = largest_object_id

    # iteratively accept objects that are close enough to an already accepted area
    continue_adding_objects = True
    while continue_adding_objects:
        continue_adding_objects = False
        for object_id, _ in other_objects:
            distance_to_largest_object = _min_distance_between_two_objects(labeled_mask, largest_object_id,
                                                                           object_id)
            if distance_to_largest_object <= max_distance:
                labeled_mask[labeled_mask == object_id] = largest_object_id
                continue_adding_objects = True

    return (labeled_mask == largest_object_id).astype(int)


def _min_distance_between_two_objects(labeled_image: np.ndarray, object1_id: int, object2_id: int) -> float:
    """
    Computes the minimal distance between two mask areas.

    `labeled_image` is a 2D array where pixels that are included in the first area have value `object1_id`
    and pixels that are included in the second area have value `object2_id`

    Returns the minimal distance between a pixel with value `object1_id` to a pixel with value `object2_id`.
    """
    object1_image = labeled_image == object1_id
    object2_image = labeled_image == object2_id

    points_coordinates_object1 = np.transpose(np.nonzero(object1_image))
    points_coordinates_object2 = np.transpose(np.nonzero(object2_image))

    distances = scipy.spatial.KDTree(points_coordinates_object1).query(points_coordinates_object2, k=1)[0]
    return min(distances, default=math.inf)


def remove_small_areas(mask: np.ndarray, max_pixel_count: int) -> np.ndarray:
    """
    Removes isolated areas of `mask` whose size is smaller or equal to `max_pixel_count`.

    Returns the resulting mask. The original mask is not modified.
    """
    mask_bool = mask.astype(bool)
    mask_bool_without_isolated_pixels = skimage.morphology.remove_small_objects(mask_bool, connectivity=2,
                                                                                min_size=max_pixel_count + 1)
    return mask_bool_without_isolated_pixels.astype(int)
