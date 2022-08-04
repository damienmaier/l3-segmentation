import pathlib

import numpy as np


def load_data_from_bm_segmenter_project(project_path: pathlib.Path):
    project_elements = [_ProjectElement(project_path, image_directory_path.name)
                        for image_directory_path in (project_path / "dicoms").iterdir()]

    project_elements_with_correct_shape = [project_element for project_element in project_elements if
                                           project_element.has_correct_shape()]
    images = [project_element.image() for project_element in project_elements_with_correct_shape]
    predicted_masks = [project_element.ml_mask() for project_element in project_elements_with_correct_shape]

    return np.array(images), np.array(predicted_masks)


class _ProjectElement:
    def __init__(self, project_path: pathlib.Path, image_name: str) -> None:
        mask_file_path = project_path / "masks" / "sma" / (image_name + ".npz")
        self.mask_file_data = np.load(mask_file_path, allow_pickle=True)

        image_file_path = project_path / "dicoms" / image_name / "0.npz"
        self.image_file_data = np.load(image_file_path, allow_pickle=True)

    def ml_mask(self):
        return self.mask_file_data["predicted"]

    def image(self):
        return self.image_file_data["matrix"]

    def has_correct_shape(self):
        return self.image().shape == (512, 512)
