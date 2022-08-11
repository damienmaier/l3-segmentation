import pathlib

import numpy as np


class ProjectElement:
    ML_MASK_KEY = "predicted"
    IMAGE_KEY = "matrix"

    @staticmethod
    def get_elements_of_project(project_path: pathlib.Path, mask_name: str) -> list["ProjectElement"]:
        project_elements = [ProjectElement(project_path, image_directory_path.name, mask_name)
                            for image_directory_path in (project_path / "dicoms").iterdir()]
        return project_elements

    def __init__(self, project_path: pathlib.Path, image_name: str, mask_name: str) -> None:
        self._mask_file_path = project_path / "masks" / mask_name / (image_name + ".npz")
        self._mask_file_data = dict(np.load(self._mask_file_path, allow_pickle=True))

        self._image_file_path = project_path / "dicoms" / image_name / "0.npz"
        self._image_file_data = dict(np.load(self._image_file_path, allow_pickle=True))

    def get_machine_learning_mask(self) -> np.ndarray:
        return self._mask_file_data[self.ML_MASK_KEY]

    def set_machine_learning_mask(self, mask: np.ndarray):
        self._mask_file_data[self.ML_MASK_KEY] = mask.astype(np.uint8)
        self._update_mask_file()

    def get_image(self) -> np.ndarray:
        image = self._image_file_data[self.IMAGE_KEY]
        image_ugly_fix = np.clip(image, a_min=-1024, a_max=None)
        return image_ugly_fix

    def _update_mask_file(self):
        np.savez(self._mask_file_path, **self._mask_file_data)
