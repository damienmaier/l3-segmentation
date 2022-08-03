import collections
import math

import numpy as np
import scipy.spatial
import skimage.measure

# those values have been empirically found to give good results
MAX_DISTANCE = 10
MIN_RATIO = .1


def remove_isolated_areas(mask):
    labeled_mask, objects_count = skimage.measure.label(mask, return_num=True)

    pixels_counter = collections.Counter(labeled_mask.flat)
    largest_objects = pixels_counter.most_common()
    # the most common value is 0, so the id of the largest object is the second most common value
    largest_object_id, largest_object_size = largest_objects[1]
    other_objects = largest_objects[2:]

    # to accept an object, we give it the value largest_object_id

    # accept big objects
    for object_id, object_size in other_objects:
        if object_size >= MIN_RATIO * largest_object_size:
            labeled_mask[labeled_mask == object_id] = largest_object_id

    # iteratively accept objects that are close enough to an already accepted object
    continue_adding_objects = True
    while continue_adding_objects:
        continue_adding_objects = False
        for object_id, _ in other_objects:
            distance_to_largest_object = _min_distance_between_two_objects(labeled_mask, largest_object_id,
                                                                           object_id)
            if distance_to_largest_object <= MAX_DISTANCE:
                labeled_mask[labeled_mask == object_id] = largest_object_id
                continue_adding_objects = True

    return (labeled_mask == largest_object_id).astype(int)


def _min_distance_between_two_objects(labeled_image, object1_id, object2_id):
    object1_image = labeled_image == object1_id
    object2_image = labeled_image == object2_id

    points_coordinates_object1 = np.transpose(np.nonzero(object1_image))
    points_coordinates_object2 = np.transpose(np.nonzero(object2_image))

    distances = scipy.spatial.KDTree(points_coordinates_object1).query(points_coordinates_object2, k=1)[0]
    return math.inf if len(distances) == 0 else min(distances)
