import torch
import torch.nn as nn
from open_clip import create_model_from_pretrained
from timm.models.convnext import ConvNeXt
import torch
from torch import nn
import torch.nn.functional as F
from llava.model.multimodal_encoder.base_encoder import BaseVisionTower, ProcessorWrapper


def extract_res_interp(model_name):
    """
    Extract the base model name, image resolution, and interpolation size from the model name string.

    Args:
        model_name (str): The name of the model in the format "base_model_name-resXXX-interpYYY".

    Returns:
        tuple: A tuple containing the base model name, image resolution, and interpolation size.
    """
    base_model_name = "hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"
    res = None
    interp = None

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class CLIPConvNextTower(BaseVisionTower):

    def _post_init(self):
        # extract image resolution from model name
        if not self.delay_load:
            self.load_model()

    def __init__(self, vision_tower, args, delay_load=False):
        """
        Initialize the CLIPConvNextTower.

        Args:
            vision_tower (str): The name of the vision tower model in the format "clip-convnext-resXXX-interpYYY".
            args (argparse.Namespace): The arguments parsed from the command line.
            delay_load (bool, optional): Whether to delay loading the model. Defaults to False.
        """
        super().__init__(vision_tower, args, delay_load)

        

        base_model_name, res, interp = extract_res_interp(vision_tower)

        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp  # default 256
        self._reduction = 32

        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.unfreeze_mm_vision_tower = getattr(args, 'unfreeze_mm_vision_tower', False)
        self.is_loaded = False

        if not delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1536

    def load_model(self, device_map=None):
        """
        Load the CLIP-ConvNext model.
        """
        assert "clip-convnext" in self.vision_tower_name.lower()
        self.vision_model = "convnext"

        base_model_name, res, interp = extract_res_interp(self.vision_tower_name)

        print("This is what I got:", base_model_name, res, interp)

        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 1024
        self._interp_size = interp if interp is not None else 576 # default 256
        self._reduction = 32


        clip_model, processor = create_model_from_pretrained(self.vision_tower_name)
        processor.transforms[0].size = self._image_size
        processor.transforms[1].size = (self._image_size, self._image_size)
        self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)
        self.vision_tower: ConvNeXt = clip_model.visual.trunk
        self.vision_tower.output_tokens = True
        self._hidden_size = 1536
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    def interpolate(self, image_forward_outs):
        """
        Interpolate the image features to the desired number of patches.

        Args:
            image_forward_outs (torch.Tensor): The output features from the vision tower.

        Returns:
            torch.Tensor: The interpolated image features.
        """
        if self._interp_size is None:
            return image_forward_outs

        image_features = F.interpolate(
            image_forward_outs.float(),
            size=(self.num_patches_per_side, self.num_patches_per_side),
            mode='bilinear',
            align_corners=False
        ).to(dtype=image_forward_outs.dtype)
        image_features = image_features.flatten(2, 3).permute(0, 2, 1).contiguous()
        return image_features

    def _forward(self, images):
        """
        Perform the forward pass of the CLIPConvNextTower.

        Args:
            images (torch.Tensor): The input images.

        Returns:
            torch.Tensor: The output features from the vision tower after interpolation.
        """
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            print(image_forward_outs.shape)
            
            image_features = self.interpolate(image_forward_outs)
            print(image_features.shape)
            return image_features

    @property
    def image_size(self):
        return self._image_size

    @property
    def num_patches_per_side(self):
        """
        Get the number of patches per side.

        Returns:
            int: The number of patches per side.
        """
        if self._interp_size is None:
            return self._image_size // self._reduction
        else:
            return int(self._interp_size ** 0.5)

    @property
    def num_patches(self):
        """
        Get the total number of patches.

        Default: 256

        Returns:
            int: The total number of patches.
        """
        if self._interp_size is None:
            return (self._image_size // self._reduction) ** 2
        else:
            return self._interp_size