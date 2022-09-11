"""
Describes several model architectures for image segmentation

Each element of `architectures.architecture_builders` is a function that returns a model.

Each architecture in this package has the following input and output format :

The inputs are images with shape (<batch size>, 512, 512, 1). The extra 1 dimension is useless but a lot of tf functions
for processing images expect a (512, 512, 1) shape, so following this convention makes things simpler.

The outputs are tensors of shape (<batch size>, 512, 512, 1), where each element has a value between 0 and 1
and represents a probability for this pixel to be included in the mask.
"""

import architectures._deep_lab_v3
import architectures._unet

architecture_builders = {
    "unet": _unet.model_sma_detection,
    "deeplabv3": _deep_lab_v3.deep_lab_v3_plus
}
