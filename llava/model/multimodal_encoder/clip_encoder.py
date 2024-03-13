import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from open_clip import create_model_from_pretrained, get_tokenizer 

class ProcessorWrapper:
    def __init__(self, transform, height=378, width=378):
        self._crop_size = {
            "height": 378,
            "width": 378,
        }
        self._transforms = transform
        #print(transform)
        self.image_mean = [0.48145466, 0.4578275, 0.40821073]

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors='pt'):
        # Ensure image is a PIL Image
        output = {}
        output['pixel_values'] = [self._transforms(image)]
        return output



class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        if self.vision_tower_name == "apple/DFN5B-CLIP-ViT-H-14-378":
            self.vision_model = "dfn-clip"


            clip_model, processor = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
            self.image_processor = ProcessorWrapper(processor)
            print("DFN CLIP is", self.image_processor)
            self.vision_tower = clip_model.visual
            self.vision_tower.output_tokens = True
            print(self.vision_tower)
            self._hidden_size = 1280
        elif self.vision_tower_name == "siglip/CLIP-ViT-SO400M-14-384":
            self.vision_model = "siglip"
            print("I am loading siglip")
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
            self.image_processor = ProcessorWrapper(processor)
            self.vision_tower = clip_model.visual.trunk
            #print(self.vision_tower)
            self.vision_tower.output_tokens = True
            self._hidden_size = 1152
        else:

            self.vision_model = "oai-clip"
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True


    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        
        # Very Important for TorchXLA
        #self.vision_tower.vision_model.encoder.gradient_checkpointing = False

        if self.vision_model == "oai-clip":


            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint 
        
        

        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            if self.vision_model == "oai-clip":
                with torch.no_grad():
                    image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
                    image_features = self.feature_select(image_forward_outs).to(images.dtype)
            elif self.vision_model == "dfn-clip":
                with torch.no_grad():
                    #print(images.shape)
                    _, image_features = self.vision_tower(images.to(device=self.device, dtype=self.dtype))
            elif self.vision_model == "siglip":
                with torch.no_grad():
                    #print(images.shape)
                    image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
                    #print(image_forward_outs.shape)
                    image_features = self.feature_select(image_forward_outs).to(images.dtype)


        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        # Dynamically infer the dtype from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, 'dtype'):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return params[0].dtype if len(params) > 0 else torch.float32  # Default to torch.float32 if no parameters

    @property
    def device(self):
        # Dynamically infer the device from the first parameter, if not explicitly specified
        if hasattr(self.vision_tower, 'device'):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return params[0].device if len(params) > 0 else torch.device("cpu")  # Default to CPU if no parameters
    


    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        
        try:
            return self.config.hidden_size
        except:
            return self._hidden_size
    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
