import _unet
import _deep_lab_v3

architecture_builders = {
    "unet": _unet.model_sma_detection,
    "deeplabv3": _deep_lab_v3
}
