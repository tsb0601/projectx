import torch

from ezcolorlog import root_logger as logger

from transformers import Dinov2Model, AutoImageProcessor, Dinov2Config

from .base_encoder import BaseVisionTower


class DinoVisionTower(BaseVisionTower):
    def _post_init(self):
        """try to extract an image resolution from the model name
        
        valid model names:
            facebook/dinov2-small
            facebook/dinov2-base
            facebook/dinov2-large
            facebook/dinov2-giant
            facebook/dinov2-giant-imagenet1k-1-layer
        
        res pattern: <model_name>-res<res>

        eg: facebook/dinov2-small-res224
        """

        # extract image resolution from model name
        if self.vision_tower_name.startswith("facebook/dinov2-"):
            parts = self.vision_tower_name.split("-res")
            if len(parts) == 2:
                self._image_size = int(parts[1])
                logger.warning(f"Extracted image size {self._image_size} from passed model name '{self.vision_tower_name}' (using {parts[0]})")
                self.vision_tower_name = parts[0]
            else:
                self._image_size = None


        if not self.delay_load:
            self.load_model()
        else:
            self.cfg_only = Dinov2Config.from_pretrained(self.vision_tower_name)

    def load_model(self):

        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name)

        _image_size = self.vision_tower.config.image_size
        if self._image_size is None:
            self._image_size = _image_size
        else:
            logger.warning(f"Overriding DinoVisionTower image size of {_image_size} with {self._image_size}")

        self.image_processor = AutoImageProcessor.from_pretrained(self.vision_tower_name, crop_size=dict(height=self._image_size, width=self._image_size))
        #print(self.vision_tower)
        # Assign the output channels of the projection convolution as the hidden size
        self._hidden_size = self.vision_tower.embeddings.patch_embeddings.projection.out_channels
        # Assign the first value of the stride of the projection convolution as the patch size
        self._patch_size = self.vision_tower.embeddings.patch_embeddings.projection.stride[0]

        #print(self._hidden_size, self._patch_size)

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True
    
    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs["last_hidden_state"]  # batch_size, sequence_length, hidden_size
        
        if self.select_feature == 'cls_patch':
            image_features = sequence_output
        elif self.select_feature == 'patch':
            image_features = sequence_output[:, 1:]
        elif self.select_feature == 'cls':
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            return image_features
