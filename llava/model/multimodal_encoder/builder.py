from ezcolorlog import root_logger as logger

from .load import load_vision_model
from .hybrid_encoder import HybridVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower is None or not isinstance(vision_tower, str):
        raise ValueError(f'Vision Tower is not specified in the config: {vision_tower_cfg}')

    if vision_tower.lower().startswith("hybridmodel"):
        logger.info(f"Loading **Hybrid** Vision Tower: {vision_tower}")
        return HybridVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    return load_vision_model(vision_tower, args=vision_tower_cfg, **kwargs)
