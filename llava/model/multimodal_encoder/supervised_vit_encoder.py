import torch
from torch import nn
import numpy as np
from torch_xla.utils.checkpoint import checkpoint
from huggingface_hub import hf_hub_download
import torch.nn.functional as F
from .base_encoder import ProcessorWrapper

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from .base_encoder import BaseVisionTower
import timm


from .ijepa.vision_transformer import vit_huge, vit_giant

from torchvision import transforms


class SupervisedViT_VisionTower(BaseVisionTower):

    def _post_init(self):
        # extract image resolution from model name
        if self.vision_tower_name.startswith("mae"):
            self._image_size = 224

        if not self.delay_load:
            self.load_model()


    def load_model(self):
        self.vision_model = "supervised"

        if self.vision_tower_name.lower()=="supervised-vit-h-14-in21k":
            self.vision_tower = timm.create_model('vit_huge_patch14_224.orig_in21k', pretrained=True)
        elif self.vision_tower_name.lower()=="supervised-vit-l-16-in21k":
            self.vision_tower = timm.create_model('vit_large_patch16_224.orig_in21k', pretrained=True)
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        #print(self.vision_tower)
        self.vision_tower.output_tokens = True
        
        self._hidden_size = self.vision_tower.embed_dim
        self._image_size = self.vision_tower.patch_embed.img_size[0]
        self._patch_size = self.vision_tower.patch_embed.patch_size[0]
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.vision_tower)
        transforms = timm.data.create_transform(**data_config, is_training=False)

        self.image_processor = ProcessorWrapper(transforms, height=self._image_size, width=self._image_size)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))[:, 1:, :]
            
            return image_features
    @property
    def patch_size(self):
        return self._patch_size 
    
    @property
    def image_size(self):
        return self._image_size
    
    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2

