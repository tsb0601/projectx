import torch
from torch import nn
import numpy as np
from torch_xla.utils.checkpoint import checkpoint
from huggingface_hub import hf_hub_download
import torch.nn.functional as F

from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from .base_encoder import BaseVisionTower
import timm

from transformers import DPTImageProcessor, DPTForDepthEstimation

from torchvision import transforms

class ProcessorWrapper:
    def __init__(self, transform, height=378, width=378, image_mean = [0.48145466, 0.4578275, 0.40821073]):
        self._crop_size = {
            "height": height,
            "width": width, 
        }
        self._transforms = transform
        self.to_tensor = transforms.Compose([
            # Add any additional transformations here (e.g., Resize, Normalize)
            transforms.ToTensor(),  # This converts the PIL Image to a PyTorch Tensor
        ])
        #print(transform)
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors='pt'):
        # Ensure image is a PIL Image
        output = self._transforms(images = image, return_tensors="pt")

        
        # Convert the NumPy array to a PyTorch tensor
        #output['pixel_values'] = [torch.from_numpy(output['pixel_values'][0])]
        
        return output

class MiDaSVisionTower(BaseVisionTower):

    def _post_init(self):
        # extract image resolution from model name
        if self.vision_tower_name.startswith("midas"):
            self._image_size = 384

        if not self.delay_load:
            print("Use this?")
            self.load_model()


    def load_model(self):
        self.vision_model = "midas"

        if self.vision_tower_name.lower()=="hybrid-midas":
            transforms = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            self.vision_tower = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").dpt
            self._hidden_size = 768
        elif self.vision_tower_name.lower()=="large-midas":
            transforms = DPTImageProcessor.from_pretrained("Intel/dpt-large")
            self.vision_tower = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").dpt
            self._hidden_size = 1024

        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}')

        #print(self.vision_tower)
        self.vision_tower.output_tokens = True
        
        self._image_size = 384
        self._patch_size = 16
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(self.vision_tower)

        self.image_processor = ProcessorWrapper(transforms, height=self._image_size, width=self._image_size, image_mean = [0.5,0.5,0.5])


        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            
            image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype)).last_hidden_state
            image_features = image_features[:, 1:, :]
            #print(image_features.shapes)
            
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

