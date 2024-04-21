import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from open_clip import create_model_from_pretrained, get_tokenizer 

from .base_encoder import BaseVisionTower


class ClipVisionTower(BaseVisionTower):
    def _post_init(self):
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

        # Very Important for TorchXLA
        from torch_xla.utils.checkpoint import checkpoint
        self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint 

    def _feature_select(self, image_features):
        if self.select_feature == 'patch':
            features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return features

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return self._feature_select(image_features)

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            return image_features
