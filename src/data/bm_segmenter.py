import pathlib

import numpy as np


class ProjectElement:
    ML_MASK_KEY = "predicted"
    IMAGE_KEY = "matrix"

    @staticmethod
    def get_elements_of_project(project_path: pathlib.Path) -> list["ProjectElement"]:
        project_elements = [ProjectElement(project_path, image_directory_path.name)
                            for image_directory_path in (project_path / "dicoms").iterdir()]
        return project_elements

    def __init__(self, project_path: pathlib.Path, image_name: str) -> None:
        self.mask_file_path = project_path / "masks" / "sma" / (image_name + ".npz")
        self.mask_file_data = np.load(self.mask_file_path, allow_pickle=True)

        self.image_file_path = project_path / "dicoms" / image_name / "0.npz"
        self.image_file_data = np.load(self.image_file_path, allow_pickle=True)

    def get_machine_learning_mask(self) -> np.ndarray:
        return self.mask_file_data[self.ML_MASK_KEY]

    def set_machine_learning_mask(self, mask: np.ndarray):
        self.mask_file_data[self.ML_MASK_KEY] = mask
        self._update_mask_file()

    def get_image(self) -> np.ndarray:
        return self.image_file_data[self.IMAGE_KEY]

    def _update_mask_file(self):
        np.savez(self.mask_file_path, self.mask_file_data)
