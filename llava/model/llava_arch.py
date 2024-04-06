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
import torch.nn.functional as F


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

class LlavaMetaForCausalLM(ABC):

	@abstractmethod
	def get_model(self):
		pass

	def get_vision_tower(self):
		return self.get_model().get_vision_tower()

	def encode_images(self, images, languages = None):
		image_features = self.get_model().get_vision_tower()(images, languages=languages)
		image_features = self.get_model().mm_projector(image_features).to(images.dtype)
		return image_features


	def prepare_inputs_labels_for_multimodal(
		self, input_ids, position_ids, attention_mask, past_key_values, labels,
		images, image_sizes=None
	):
		vision_tower = self.get_vision_tower()
		if vision_tower is None or images is None or input_ids.shape[1] == 1:
			return input_ids, position_ids, attention_mask, past_key_values, None, labels

		
		# embed the input_ids
		new_input_ids_padded_for_emb = torch.where(input_ids==IMAGE_TOKEN_INDEX, 0, input_ids)
		input_embeds = self.get_model().embed_tokens(new_input_ids_padded_for_emb)

		language_embeds = None
		
		# mask = (labels == -100) & (attention_mask) & (input_ids!=0) & (input_ids!=IMAGE_TOKEN_INDEX)
		# mask[:, :35] = False 
		# non_zero_counts = mask.sum(dim=1)

		# #print("Count of instruction entries in the mask for each sample:", non_zero_counts)

		# mask_expanded = mask.unsqueeze(-1).float()  # Adds an embedding_dim axis with size 1

		# # Multiply input_embeds by the expanded mask to zero out unwanted embeddings
		# filtered_input_embeds = input_embeds * mask_expanded

		# # Placeholder values for demonstration
		# number_of_text_tokens = 144  # The target number of tokens
		# embedding_dim = filtered_input_embeds.size(2)  # Assuming the last dimension is embedding size

		# # Compute the number of non-zero embeddings per sequence
		# non_zero_counts = mask.sum(dim=1)

		# # Initialize a tensor to hold the output
		# language_embeds = torch.zeros((filtered_input_embeds.size(0), number_of_text_tokens, embedding_dim),
		# 							device=filtered_input_embeds.device, dtype=filtered_input_embeds.dtype)

		# # Loop over batches to truncate or pad
		# # Note: This approach assumes compacting is necessary and uses a loop due to per-sequence operations.
		# for i, (embeds, count) in enumerate(zip(filtered_input_embeds, non_zero_counts)):
		# 	# Determine the end index for copying embeddings, limited by the target number or actual non-zero count
		# 	end_idx = min(count.item(), number_of_text_tokens)
		# 	if end_idx > 0:
		# 		# Copy the embeddings up to end_idx
		# 		language_embeds[i, :end_idx] = embeds[mask[i]][:end_idx]

		#print("input embed shape is", input_embeds.shape)
		#print("labels shape is", labels.shape)
		#print("attention mask shape is", attention_mask.shape)
		#print("language embed shape is", language_embeds.shape)

		########################
		### embed the images ###
		########################

		if type(images) is list or images.ndim == 5:
			if type(images) is list:
				images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
			concat_images = torch.cat([image for image in images], dim=0)
			image_features = self.encode_images(concat_images)
			split_sizes = [image.shape[0] for image in images]
			image_features = torch.split(image_features, split_sizes, dim=0)
			mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
			image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
			if mm_patch_merge_type == 'flat':
				image_features = [x.flatten(0, 1) for x in image_features]
			elif mm_patch_merge_type.startswith('spatial'):
				new_image_features = []
				for image_idx, image_feature in enumerate(image_features):
					if image_feature.shape[0] > 1:
						base_image_feature = image_feature[0]
						image_feature = image_feature[1:]
						height = width = self.get_vision_tower().num_patches_per_side
						assert height * width == base_image_feature.shape[0]
						if image_aspect_ratio == 'anyres':
							num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
							image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
						else:
							raise NotImplementedError
						if 'unpad' in mm_patch_merge_type:
							image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
							image_feature = image_feature.flatten(1, 2).flatten(2, 3)
							image_feature = unpad_image(image_feature, image_sizes[image_idx])
							image_feature = torch.cat((
								image_feature,
								self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1)
							), dim=-1)
							image_feature = image_feature.flatten(1, 2).transpose(0, 1)
						else:
							image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
							image_feature = image_feature.flatten(0, 3)
						image_feature = torch.cat((base_image_feature, image_feature), dim=0)
					else:
						image_feature = image_feature[0]
						if 'unpad' in mm_patch_merge_type:
							image_feature = torch.cat((
								image_feature,
								self.model.image_newline[None]
							), dim=0)
					new_image_features.append(image_feature)
				image_features = new_image_features
			else:
				raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
		else:
			image_features = self.encode_images(images, languages=language_embeds)

		# TODO: image start / end is not implemented here to support pretraining.
		if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
			raise NotImplementedError
		
		



		new_input_embeds = []
		cur_image_idx = 0
		# insert the image embeddings
		for batch_idx, (cur_input_embeds, cur_input_ids) in enumerate(zip(input_embeds, input_ids)):
			num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
			if num_images == 0:
				cur_image_idx += 1
				new_input_embeds.append(cur_input_embeds)
				continue

			image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

			cur_input_embeds_im_replaced = []

			prev_image_length = 0
			for i in range(len(image_token_indices) - 1):
				# skip the image tokens (1 indicator + (image_length-1) paddings)
				cur_input_embeds_im_replaced.append(cur_input_embeds[image_token_indices[i]+1+prev_image_length:image_token_indices[i+1]])
				if i < len(image_token_indices) - 2:
					cur_image_features = image_features[cur_image_idx]
					prev_image_length = len(cur_image_features)-1
					cur_image_idx += 1
					cur_input_embeds_im_replaced.append(cur_image_features)

			cur_input_embeds_im_replaced = [x.to(self.device) for x in cur_input_embeds_im_replaced]
			new_input_embeds.append(torch.cat(cur_input_embeds_im_replaced))

		new_input_embeds = torch.stack(new_input_embeds)

		return None, position_ids, attention_mask, past_key_values, new_input_embeds, labels

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
