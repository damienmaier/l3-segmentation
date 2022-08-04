import architectures._deep_lab_v3
import architectures._unet

architecture_builders = {
    "unet": _unet.model_sma_detection,
    "deeplabv3": _deep_lab_v3.deep_lab_v3_plus
}
