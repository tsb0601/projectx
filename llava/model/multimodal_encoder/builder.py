import os

from ezcolorlog import root_logger as logger

from .clip_encoder import ClipVisionTower
from .dfn_clip_encoder import DfnClipVisionTower
from .siglip_encoder import SiglipVisionTower
from .eva_clip_encoder import EvaClipVisionTower
from .dino_encoder import DinoVisionTower
from .ijepa_encoder import IJepaVisionTower
from .mae_encoder import MAEVisionTower
from .midas_encoder import MiDaSVisionTower
from .moco_encoder import MoCoVisionTower
from .hybrid_encoder import HybridVisionTower
from .supervised_vit_encoder import SupervisedViT_VisionTower
from .diffusion_encoder import DiffusionVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    if vision_tower is None or not isinstance(vision_tower, str):
        raise ValueError(f'Vision Tower is not specified in the config: {vision_tower_cfg}')

    # CLIP-based Vision Towers
    #print(vision_tower_cfg)
    if "hybridmodel" in vision_tower.lower():
        logger.info(f"Loading **Hybrid** Vision Tower: {vision_tower}")
        return HybridVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    if "openai/clip" in vision_tower.lower():
        logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower}")
        return ClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "apple/dfn" in vision_tower.lower():
        logger.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower}")
        return DfnClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "siglip" in vision_tower.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower}")
        return SiglipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "eva/clip" in vision_tower.lower():
        logger.info(f"Loading **EVA CLIP** Vision Tower: {vision_tower}")
        return EvaClipVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "diffusion" in vision_tower.lower():
        logger.info(f"Loading **Diffusion** Vision Tower: {vision_tower}")
        return DiffusionVisionTower(vision_tower, args=args)  

    if "ijepa" in vision_tower.lower():
        logger.info(f"Loading **IJepa** Vision Tower: {vision_tower}")
        return IJepaVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "mae" in vision_tower.lower():
        logger.info(f"Loading **MAE** Vision Tower: {vision_tower}")
        return MAEVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "midas" in vision_tower.lower():
        logger.info(f"Loading **MiDaS** Vision Tower: {vision_tower}")
        return MiDaSVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "moco" in vision_tower.lower():
        logger.info(f"Loading **MoCo** Vision Tower: {vision_tower}")
        return MoCoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    if "supervised-vit" in vision_tower.lower():
        logger.info(f"Loading **Supervised** Vision Tower: {vision_tower}")
        return SupervisedViT_VisionTower(vision_tower, args=vision_tower_cfg, **kwargs)     
    # dinov2
    if "dinov2" in vision_tower.lower():
        logger.info(f"Loading **DINO Vision Tower: {vision_tower}")
        return DinoVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
    
