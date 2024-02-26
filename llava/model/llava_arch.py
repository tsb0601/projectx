#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor
import time

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
       
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list:
            # Flatten the list of lists into a single list
            if not images:  # This checks if images is an empty list
                # Handle the empty list case as needed
                # For example, set image_features to None or an empty tensor
                image_features = None  # or torch.tensor([]), depending on your encode_images method's requirements
            else:
                flattened_images = [item for sublist in images for item in sublist]

                # Process the flattened list of tensors
                if type(flattened_images) is list:
                    flattened_images = [x.unsqueeze(0) if x.ndim == 3 else x for x in flattened_images]
                concat_images = torch.cat([image for image in flattened_images], dim=0)
                image_features = self.encode_images(concat_images)           
        else:
            image_features = self.encode_images(images)
            

        
        ##################################
        ## Changes Made for Multi Image ##
        ##################################
            
         # remove the padding using attention_mask -- FIXME
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
            
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
                    
        _, image_feature_size, embedding_size = image_features.shape


        # Calculate new sequence lengths after image feature insertions
        # Assuming image_feature_size is the number of tokens each image feature will occupy
        # Calculate adjusted lengths for each sequence using list comprehension
        adjusted_lengths = [
            len(sequence) - (sequence == IMAGE_TOKEN_INDEX).sum().item() + 
            (sequence == IMAGE_TOKEN_INDEX).sum().item() * image_feature_size
            for sequence in input_ids
        ]

        # Determine max sequence length after adjustments
        max_seq_length = max(adjusted_lengths)
        batch_size = len(input_ids)   

        new_input_embeds_padded = torch.zeros(batch_size, max_seq_length, embedding_size, device=_input_ids.device)
        new_labels_padded = torch.full((batch_size, max_seq_length), IGNORE_INDEX, dtype=_input_ids.dtype, device=_input_ids.device)

        # Step 2: Efficiently integrate image features
        # This step would typically require identifying positions of IMAGE_TOKEN_INDEX in each sequence
        # for batch_idx, cur_input_ids in enumerate(input_ids):
        #     seq_length = len(cur_input_ids)
        #     image_feature_index = 0  # Index to track which image feature to insert


        # Initialize a tracker for current index in image_features
        cur_image_feature_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            

            # Determine the positions of IMAGE_TOKEN_INDEX in the current sequence
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            num_images = len(image_token_indices)

            # Calculate the start and end indices for segments without IMAGE_TOKEN_INDEX
            start_indices = torch.cat([torch.tensor([0], device=cur_input_ids.device), image_token_indices + 1])
            end_indices = torch.cat([image_token_indices, torch.tensor([len(cur_input_ids)], device=cur_input_ids.device)])

            # Placeholder for the next position in new_input_embeds_padded to fill
            fill_position = 0

            for i in range(len(start_indices)):
                # Extract and embed the segment of cur_input_ids between IMAGE_TOKEN_INDEX positions
                segment = cur_input_ids[start_indices[i]:end_indices[i]]
                if len(segment) > 0:  # Check if the segment is not empty
                    segment_embeds = self.get_model().embed_tokens(segment)
                    new_input_embeds_padded[batch_idx, fill_position:fill_position + segment_embeds.size(0), :] = segment_embeds
                    new_labels_padded[batch_idx, fill_position:fill_position + len(segment)] = labels[batch_idx][start_indices[i]:end_indices[i]]
                    fill_position += segment_embeds.size(0)

                # Insert image features after each text segment, except for the last segment
                if i < num_images:
                    new_input_embeds_padded[batch_idx, fill_position:fill_position + image_feature_size, :] = image_features[cur_image_feature_idx].unsqueeze(0)
                    new_labels_padded[batch_idx, fill_position:fill_position + image_feature_size] = IGNORE_INDEX
                    fill_position += image_feature_size
                    cur_image_feature_idx += 1

         # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds_padded = new_input_embeds_padded[:, :tokenizer_model_max_length]
            new_labels_padded = new_labels_padded[:, :tokenizer_model_max_length]
        
        # new_input_embeds_padded = new_input_embeds_padded.to(dtype=)
        


        #####################
        ## Conlude Changes ##
        #####################
        max_seq_length = min(max_seq_length, tokenizer_model_max_length) if tokenizer_model_max_length is not None else max_seq_length

        attention_mask = torch.zeros((batch_size, max_seq_length), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_seq_length), dtype=position_ids.dtype, device=position_ids.device)

        # Example sequence lengths
        sequence_lengths = torch.tensor(adjusted_lengths, dtype=new_labels_padded.dtype, device=new_labels_padded.device)

        # Create a matrix of positions [batch_size, max_len]
        position_matrix = torch.arange(max_seq_length).expand(batch_size, -1).to(dtype = sequence_lengths.dtype, device=sequence_lengths.device)

        # Create attention_mask by comparing position matrix with sequence lengths
        attention_mask = (position_matrix < sequence_lengths.unsqueeze(1)).to(dtype=_attention_mask.dtype, device=_attention_mask.device)

        # Create position_ids similarly, but filled with actual positions; masked positions will not be used
        position_ids = position_matrix * attention_mask

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded


        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds_padded, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
