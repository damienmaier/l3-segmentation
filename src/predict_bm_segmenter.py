import pathlib

import numpy as np

import data.bm_segmenter
import final_model


def compute_predictions_on_bm_segmenter_project(project_path: pathlib.Path, mask_name: str):
    project_elements = data.bm_segmenter.ProjectElement.get_elements_of_project(project_path, mask_name=mask_name)
    images = [project_element.get_image() for project_element in project_elements]

    predicted_masks = final_model.predict_from_images_iterable(images)

    for project_element, predicted_mask in zip(project_elements, predicted_masks):
        project_element.set_machine_learning_mask(predicted_mask)
