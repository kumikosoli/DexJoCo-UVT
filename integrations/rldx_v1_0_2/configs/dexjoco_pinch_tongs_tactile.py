"""RLDX modality registration for the tactile DexJoCo arm."""

from dexjoco_pinch_tongs_common import build_modality_config
from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag


register_modality_config(
    build_modality_config(include_tactile=True),
    EmbodimentTag.GENERAL_EMBODIMENT,
)
