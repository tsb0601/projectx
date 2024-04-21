import torch
from torch import nn
import numpy as np
from torch_xla.utils.checkpoint import checkpoint
from huggingface_hub import hf_hub_download
import torch.nn.functional as F

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from transformers import SamModel, SamVisionConfig, SamProcessor
from transformers.models.sam.modeling_sam import SamVisionEncoder

from .base_encoder import BaseVisionTower
from .sam.transforms import ResizeLongestSide
from .sam.encoder import create_sam_vit, SAM_MODEL_CONFIG


class SamVisionTower(BaseVisionTower):
    def _post_init(self):
        if not self.delay_load:
            self.load_model()
        else:
            self.cfg_only = SamVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.vision_tower: SamVisionEncoder = SamModel.from_pretrained(self.vision_tower_name).vision_encoder
        self.image_processor = SamProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            output = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            embeddings = output[0]
            return embeddings

    @property
    def patch_size(self):
        return self.vision_tower.config.patch_size
    
    @property
    def image_size(self):
        return self.vision_tower.config.image_size
    
    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2


class ProcessorWrapper:
    def __init__(self, height=378, width=378):
        self._crop_size = {
            "height": height,
            "width": width, 
        }
        self._transforms = ResizeLongestSide((height))

        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors='pt'):
        input_image = self._transforms.apply_image(np.array(image))
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # Normalize colors
        input_image_torch = (input_image_torch - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = input_image_torch.shape[-2:]
        padh = self._crop_size['height'] - h
        padw = self._crop_size['width']  - w
        input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
        output = {}
        output['pixel_values'] = [input_image_torch[0]]
        return output


class SAMVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        assert vision_tower in SAM_MODEL_CONFIG.keys()

        self.num_patches = args.image_aux_token_len
        self.num_patches_per_side = int(self.num_patches**0.5)
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        # self.unfreeze_mm_vision_tower = False
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self._hidden_size = SAM_MODEL_CONFIG[vision_tower]['width']
            self.hidden_size_single = SAM_MODEL_CONFIG[vision_tower]['width']
            self.size = 1024

    def load_model(self):
        
        self.image_processor = ProcessorWrapper(height=1024, width=1024)
        self.vision_tower = create_sam_vit(self.vision_tower_name)
        
        self._hidden_size = SAM_MODEL_CONFIG[self.vision_tower_name]['width']
        self.hidden_size_single = SAM_MODEL_CONFIG[self.vision_tower_name]['width']
        self.size = 1024
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def forward(self, images):
        # Very Important for TorchXLA
        # self.vision_tower.vision_model.encoder.gradient_checkpointing = False
        if type(images) is list:
            image_features = []
            for image in images:
                x = self.vision_tower.patch_embed(image)
                if self.vision_tower.pos_embed is not None:
                    x = x + self.vision_tower.pos_embed

                for blk in self.vision_tower.blocks:
                    x = blk(x)
                image_feature = x
                image_features.append(image_feature)
        else:
            with torch.requires_grad(self.unfreeze_mm_vision_tower):
                ctx_manager = torch.cpu.amp.autocast(cache_enabled=True, dtype=torch.bfloat16)
                with ctx_manager:
                    x = self.vision_tower.patch_embed(images.to(device=self.device, dtype=self.dtype))
                    if self.vision_tower.pos_embed is not None:
                        x = x + self.vision_tower.pos_embed

                    for blk in self.vision_tower.blocks:
                        x = blk(x)
                    image_features = x.permute(0, 3, 1, 2).contiguous()
                    image_features_origin = image_features
                    target_h = target_w = self.num_patches_per_side
                    image_features_flatten = F.interpolate(image_features.float(), 
                                                    size=(target_h, target_w), 
                                                    mode='bilinear', 
                                                    align_corners=False).to(dtype=image_features.dtype)
                    image_features_flatten = image_features_flatten.flatten(2, 3).permute(0, 2, 1).contiguous()
                    return image_features_flatten

                # image_features += self.pos_embed[None, :, :]
        return image_features
