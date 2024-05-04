import torch
from torch import nn
import numpy as np
from torch_xla.utils.checkpoint import checkpoint
from huggingface_hub import hf_hub_download
import torch.nn.functional as F
from .base_encoder import ProcessorWrapper

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from .base_encoder import BaseVisionTower


from .ijepa.vision_transformer import vit_huge, vit_giant

from torchvision import transforms

import torch
from torchvision.utils import make_grid
from PIL import Image

def revert_preprocessing(images):
    def convert_to_pil(tensor):
        return to_pil_image(tensor)
    
    reversed_images = []
    for image in images:
        reversed_image = convert_to_pil(image)
        reversed_images.append(reversed_image)
    
    return reversed_images

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

from .supervised_vit_encoder import SupervisedViT_VisionTower

def load_vision_model(vision_tower, args):
    if "openai/clip" in vision_tower.lower():
        logger.info(f"Loading **OpenAI CLIP** Vision Tower: {vision_tower}")
        return ClipVisionTower(vision_tower, args=args)
    if "apple/dfn" in vision_tower.lower():
        logger.info(f"Loading **Apple DFN CLIP** Vision Tower: {vision_tower}")
        return DfnClipVisionTower(vision_tower, args=args)
    if "siglip" in vision_tower.lower():
        logger.info(f"Loading **SigLIP CLIP** Vision Tower: {vision_tower}")
        return SiglipVisionTower(vision_tower, args=args)
    if "eva/clip" in vision_tower.lower():
        logger.info(f"Loading **EVA CLIP** Vision Tower: {vision_tower}")
        return EvaClipVisionTower(vision_tower, args=args)
    if "ijepa" in vision_tower.lower():
        logger.info(f"Loading **IJepa** Vision Tower: {vision_tower}")
        return IJepaVisionTower(vision_tower, args=args)
    if "mae" in vision_tower.lower():
        logger.info(f"Loading **MAE** Vision Tower: {vision_tower}")
        return MAEVisionTower(vision_tower, args=args)
    if "midas" in vision_tower.lower():
        logger.info(f"Loading **MiDaS** Vision Tower: {vision_tower}")
        return MiDaSVisionTower(vision_tower, args=args)
    if "moco" in vision_tower.lower():
        logger.info(f"Loading **MoCo** Vision Tower: {vision_tower}")
        return MoCoVisionTower(vision_tower, args=args)
    if "supervised-vit" in vision_tower.lower():
        logger.info(f"Loading **Supervised** Vision Tower: {vision_tower}")
        return SupervisedViT_VisionTower(vision_tower, args=args)     
    # dinov2
    if "dinov2" in vision_tower.lower():
        logger.info(f"Loading **DINO Vision Tower: {vision_tower}")
        return DinoVisionTower(vision_tower, args=args)

class HybridVisionTower(BaseVisionTower):

    def _post_init(self):
        # extract image resolution from model name

        model_names = self.vision_tower_name.split("-and-")
        model_names[0] = model_names[0].replace("hybridmodel-", "")
        self.model_names = model_names

        print(self.model_names)


        if not self.delay_load:
            self.load_model()


    def load_model(self):
        self.vision_model = "hybrid"
        for i, model_name in enumerate(self.model_names, start=1):
            setattr(self, f"vision_tower_{i}", load_vision_model(model_name, args=self.args))

        self._hidden_size = sum([getattr(self, f"vision_tower_{i}")._hidden_size for i in range(1, len(self.model_names) + 1)])
        self._image_size = 384
        self._patch_size = 16

        # preprocess = transforms.Compose([
        #     transforms.Resize(size=(384, 384), interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # ])
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        

        self.image_processor = ProcessorWrapper(preprocess, height=self._image_size, width=self._image_size)

        for i in range(1, len(self.model_names) + 1):
            getattr(self, f"vision_tower_{i}").requires_grad_(self.unfreeze_mm_vision_tower)

        self.is_loaded = True

    def forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            raw_images = revert_preprocessing(images)

            output_images_features = []
            for i in range(1, len(self.model_names) + 1):
                vision_tower = getattr(self, f"vision_tower_{i}")
                processed_images = [image[i-1] for image in images]
                batch_tensor = torch.stack(processed_images)
                #print("batch tensor", batch_tensor.shape)
                image_features = vision_tower._forward(batch_tensor.to(device=self.device, dtype=self.dtype))

                b, num_tokens, dim = image_features.shape
                if num_tokens != self.image_token_len:
                    target_h = target_w = int(np.sqrt(self.image_token_len))
                    h = w = int(np.sqrt(num_tokens))
                    image_features = image_features.view(b, h, w, dim)
                    image_features = image_features.permute(0, 3, 1, 2).contiguous()
                    image_features = F.interpolate(image_features.to(torch.float32), size=(target_h, target_w), mode='bilinear', align_corners=False).to(image_features.dtype)
                    image_features = image_features.permute(0, 2, 3, 1).contiguous().flatten(1, 2)


                output_images_features.append(image_features)

            output_tensor = torch.cat(output_images_features, dim=-1)
            #print("output", output_tensor.shape)

            return output_tensor

    @property
    def patch_size(self):
        return self._patch_size 
    
    @property
    def image_size(self):
        return self._image_size
    
    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower_1, 'dtype'):
            return self.vision_tower_1.dtype
        else:
            params = list(self.vision_tower_1.parameters())
            return params[0].dtype if len(params) > 0 else torch.float32  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower_1, 'device'):
            return self.vision_tower_1.device
        else:
            params = list(self.vision_tower_1.parameters())
            return params[0].device if len(params) > 0 else torch.device("cpu")  # Default to CPU if no parameters


