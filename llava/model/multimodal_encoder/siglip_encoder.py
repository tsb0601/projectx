import torch
from open_clip import create_model_from_pretrained 

from .base_encoder import ProcessorWrapper
from .clip_encoder import ClipVisionTower

import numpy as np

class SiglipVisionTower(ClipVisionTower):
    def load_model(self):
        self.vision_model = "siglip"
        if self.vision_tower_name in ("siglip/CLIP-ViT-SO400M-14-384", "timm/ViT-SO400M-14-SigLIP-384"):
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
            # self.image_processor = ProcessorWrapper(processor,height=384, width=384)
        elif self.vision_tower_name in ("timm/ViT-SO400M-14-SigLIP", "siglip/CLIP-ViT-SO400M-14"):
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP')
            # self.image_processor = ProcessorWrapper(processor, height=224, width=224)
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')
        
        print(processor)
        self.vision_tower = clip_model.visual.trunk
        self.vision_tower.output_tokens = True

        self._hidden_size = self.vision_tower.embed_dim
        self._image_size = self.vision_tower.patch_embed.img_size[0]
        self._patch_size = self.vision_tower.patch_embed.patch_size[0]
        self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _forward(self, images, interpolate_token = 576):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))

            return image_features