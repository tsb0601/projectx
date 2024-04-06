import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from open_clip import create_model_from_pretrained, get_tokenizer 




from einops import rearrange, repeat

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        
        b, h = x.shape[0], self.heads
        
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)
        
        q, k, v = rearrange(q, 'b n (h d) -> b h n d', h = h), rearrange(k, 'b n (h d) -> b h n d', h = h), rearrange(v, 'b n (h d) -> b h n d', h = h)
        q = q * self.scale
        
        sim = torch.einsum('bhid,bhjd->bhij', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim = 1024,
        depth = 2,
        dim_head = 96,
        heads = 8,
        num_latents = 144,
        ff_mult = 4,
        language_token_dim = 4096
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_pos_emb = nn.Parameter(torch.randn(1, dim))
        self.language_cross_attn = nn.MultiheadAttention(dim, heads)
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))
        
        self.norm = nn.LayerNorm(dim)
        self.language_proj = nn.Linear(language_token_dim, dim)

    def forward(self, x, language_tokens):
        if x.ndim == 2:
            x = rearrange(x, 'b d -> b 1 d')
        print("x.shape", x.shape)
        print("mdeia_pos_emb.shape", self.media_pos_emb.shape)
        x = x + self.media_pos_emb
        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])
        
        language_tokens = self.language_proj(language_tokens)
        
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            print("latents.shape", latents.shape)
            print("language_tokens.shape", language_tokens.shape)
            latents = self.language_cross_attn(latents, language_tokens, language_tokens)[0] + latents
            latents = ff(latents) + latents
        
        return self.norm(latents)
    





class ProcessorWrapper:
    def __init__(self, transform, height=378, width=378):
        self._crop_size = {
            "height": height,
            "width": width, 
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

        #self.resampler = PerceiverResampler()

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
            #print("DFN CLIP is", self.image_processor)
            self.vision_tower = clip_model.visual
            self.vision_tower.output_tokens = True
            #print(self.vision_tower)
            self._hidden_size = 1280
        elif self.vision_tower_name == "siglip/CLIP-ViT-SO400M-14-384":
            self.vision_model = "siglip"
            #print("I am loading siglip")
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
            self.image_processor = ProcessorWrapper(processor,height=384, width=384)
            self.vision_tower = clip_model.visual.trunk
            #print(self.vision_tower)
            self.vision_tower.output_tokens = True
            self._hidden_size = 1152
        elif self.vision_tower_name == "eva/CLIP-ViT-L-336":
            self.vision_model = "evaclip"
            #print("I am loading siglip")
            clip_model, processor = create_model_from_pretrained('hf-hub:timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k')
            self.image_processor = ProcessorWrapper(processor,height=336, width=336)
            self.vision_tower = clip_model.visual.trunk
            #print(self.vision_tower)
            self.vision_tower.output_tokens = True
            self._hidden_size = 1024
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
    def forward_features(self, images):
        
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
            elif self.vision_model == "siglip" or self.vision_model == "evaclip":
                with torch.no_grad():
                    #print(images.shape)
                    image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
                    #print(image_forward_outs.shape)
                    #print(image_forward_outs.shape)
                    # image_features = image_forward_outs[:, 1:]
                    image_features = image_forward_outs
        return image_features
    

    def forward(self, images, languages = None):
        image_features = self.forward_features(images)
        if languages is not None:
            image_features = self.resampler(image_features, languages)

        print("output shape", image_features.shape)
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
