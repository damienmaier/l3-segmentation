import architectures._unet
import architectures._deep_lab_v3

architecture_builders = {
    "unet": _unet.model_sma_detection,
    "deeplabv3": _deep_lab_v3.deep_lab_v3_plus
}
