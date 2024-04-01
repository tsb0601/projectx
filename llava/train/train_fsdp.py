# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
from torch import nn

import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.train.gcloud_rsync_callback import GCloudRsyncCallback

from PIL import Image

from packaging import version

from ezcolorlog import root_logger as logger, log_stdout


logger.setLevel(logging.WARNING)

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_token_len: int = 576
    image_position: int = 35


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    # GCSFS
    gcp_project: Optional[str] = field(default=None)
    """Can also set GCP_PROJECT environment variable."""
    gcs_output_dir: Optional[str] = field(default=None)
    """gs://<bucket>/<prefix>"""


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: v.cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            trainer.model.config.save_pretrained(output_dir)
        import torch_xla.core.xla_model as xm
        ckpt_prefix = os.path.join(output_dir, "mm_projector")
        
        os.makedirs(output_dir, exist_ok=True)
        rank = xm.get_ordinal()
        print(rank)
        world_size = xm.xrt_world_size()
        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'
        ckpt = {
            'model': weight_to_save,
            'shard_metadata': trainer.model.get_shard_metadata()
        }
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        xm.save(ckpt, ckpt_path, master_only=False)
        print(f'checkpoint saved to {ckpt_path}\n', end='')
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return


    trainer._save(output_dir)
    # state_dict = trainer.model.state_dict()
    # if trainer.args.should_save:
    #     cpu_state_dict = {
    #         key: value.cpu()
    #         for key, value in state_dict.items()
    #     }
    #     del state_dict
    #     trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_cohere(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    print("I am processing cohere way!!")

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        #print("tokenizer id is", tokenizer.pad_token_id)
        #print(target[:10])
        #print(target[-10:])
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        rank0_print("rounds", rounds)
        
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        rank0_print("cur_len", cur_len, "total_len", total_len)

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    print("Using v1!!!")
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        print("tokenizer id is", tokenizer.pad_token_id)
        #print(target[:10])
        #print(target[-10:])
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        print("target vs total_len", len(target), total_len)

        rounds = conversation.split(conv.sep2)

        print(rounds)

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            print("---------------")
            print("part 0 and sep", parts[0], sep)
            print("---------------")
            
            parts[0] += sep


            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1
            # if i != 0 and not getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
            #     print("I am adding one")
            #     round_len += 1
            #     instruction_len += 1

            print(f"Round {i+1}: rou length = {len(rou)}, sround_len = {round_len}, instruction_len = {instruction_len}")

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        #rank0_print("cur_len", cur_len, "total_len", total_len)

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    print("Cohere version is", conversation_lib.default_conversation.version)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("coherev1"):
        return preprocess_cohere(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        #list_data_dict = json.load(open(data_path, "r"))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        #self.list_data_dict = list_data_dict
        self.data_path = data_path
        self.data_args = data_args
        self.length = self._get_length()
        rank0_print(f"Initialized dataset with {self.length} samples from {data_path}... DataArgs: {data_args}")

    def _get_length(self):
        """Calculates the number of samples in the .jsonl file."""
        with open(self.data_path, 'r') as file:
            for i, _ in enumerate(file):
                pass
        return i + 1

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.length

    # def __len__(self):
    #     return len(self.list_data_dict)

    def _compute_lengths(self):
        """Compute and cache lengths of conversations in the dataset."""
        if hasattr(self, 'length_list') and hasattr(self, 'modality_length_list'):
            # Return cached values if already computed
            return self.length_list, self.modality_length_list

        self.length_list = []
        self.modality_length_list = []
        with open(self.data_path, 'r') as file:
            for line in file:
                sample = json.loads(line.strip())
                img_tokens = 128 if 'image' in sample else 0
                cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
                self.length_list.append(cur_len + img_tokens)
                modality_len = cur_len if 'image' in sample else -cur_len
                self.modality_length_list.append(modality_len)
        return self.length_list, self.modality_length_list

    @property
    def lengths(self):
        length_list, _ = self._compute_lengths()
        return length_list

    @property
    def modality_lengths(self):
        _, modality_length_list = self._compute_lengths()
        return modality_length_list

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = 128 if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    # @property
    # def modality_lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
    #         cur_len = cur_len if 'image' in sample else -cur_len
    #         length_list.append(cur_len)
    #     return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #sources = self.list_data_dict[i]

        with open(self.data_path, 'r') as file:
            for idx, line in enumerate(file):
                if idx == i:
                    sources = json.loads(line.strip())
                    break

        dat = sources
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = dat['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in dat))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in dat:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict



# class LazySupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str,
#                  tokenizer: transformers.PreTrainedTokenizer,
#                  data_args: DataArguments):
#         super(LazySupervisedDataset, self).__init__()
#         self.data_path = data_path
#         #list_data_dict = json.load(open(data_path, "r"))
#         rank0_print("Formatting inputs...Skip in lazy mode")
#         self.tokenizer = tokenizer
#         #self.list_data_dict = list_data_dict
#         self.data_args = data_args
#         self._setup()

#     def _setup(self):
#         self.offsets = []  # Store the offset of each line in the file for direct access
#         with open(self.data_path, 'rb') as f:
#             offset = 0
#             for line in f:
#                 self.offsets.append(offset)
#                 offset += len(line)
#         self.length = len(self.offsets)


#     def __len__(self):
#         return self.length

#     def _get_line(self, idx):
#         with open(self.data_path, 'r') as f:
#             f.seek(self.offsets[idx])
#             line = f.readline()
#             return json.loads(line)


#     @property
#     def lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             img_tokens = 128 if 'image' in sample else 0
#             length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
#         return length_list

#     @property
#     def modality_lengths(self):
#         length_list = []
#         for sample in self.list_data_dict:
#             cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
#             cur_len = cur_len if 'image' in sample else -cur_len
#             length_list.append(cur_len)
#         return length_list

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         # sources = self.list_data_dict[i]
#         # if isinstance(i, int):
#         #     sources = [sources]
#         # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

#         # Load data item by index using the lazy loading mechanism
#         data_item = self._get_line(i)

#         if 'image' in data_item:
#             image_file = data_item['image']
#             image_folder = self.data_args.image_folder
#             processor = self.data_args.image_processor
#             image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
#             if self.data_args.image_aspect_ratio == 'pad':
#                 def expand2square(pil_img, background_color):
#                     width, height = pil_img.size
#                     if width == height:
#                         return pil_img
#                     elif width > height:
#                         result = Image.new(pil_img.mode, (width, width), background_color)
#                         result.paste(pil_img, (0, (width - height) // 2))
#                         return result
#                     else:
#                         result = Image.new(pil_img.mode, (height, height), background_color)
#                         result.paste(pil_img, ((height - width) // 2, 0))
#                         return result
#                 image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             else:
#                 image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
#             sources = preprocess_multimodal(
#                 copy.deepcopy([e["conversations"] for e in sources]),
#                 self.data_args)
#         else:
#             sources = copy.deepcopy([e["conversations"] for e in sources])
#         data_dict = preprocess(
#             sources,
#             self.tokenizer,
#             has_image=('image' in data_item))
#         if isinstance(i, int):
#             data_dict = dict(input_ids=data_dict["input_ids"][0],
#                              labels=data_dict["labels"][0])

#         # image exist in the data
#         if 'image' in self.list_data_dict[i]:
#             data_dict['image'] = image
#         elif self.data_args.is_multimodal:
#             # image does not exist in the data, but the model is multimodal
#             crop_size = self.data_args.image_processor.crop_size
#             data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
#         return data_dict


def prepare_multimodal_data(input_ids, labels, attention_mask, image_token_len=576, max_length=2048):
    input_ids_im_replaced = []
    labels_im_replaced = []
    attention_mask_im_replaced = []
    position_ids_im_replaced = []
    # insert the padding tokens to the places of image so we can embed them together
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

        cur_input_ids_im_replaced = []
        cur_labels_im_replaced = []
        cur_attention_mask_im_replaced = []
        cur_position_ids_im_replaced = []
        
        cur_labels = labels[batch_idx]
        cur_attention_mask = attention_mask[batch_idx]
        index = 0
        for i in range(len(image_token_indices) - 1):
            # still keep the first image token in input_ids for further use
            cur_input_ids_im_replaced.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]+1])
            cur_labels_im_replaced.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_attention_mask_im_replaced.append(cur_attention_mask[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_position_ids_im_replaced.append(torch.arange(index, index+image_token_indices[i+1]-(image_token_indices[i]+1), dtype=torch.long, device=cur_input_ids.device))
            index += image_token_indices[i+1]-(image_token_indices[i]+1)
            
            if i < len(image_token_indices) - 2:
                cur_input_ids_im_replaced.append(torch.full((image_token_len-1,), 0, device=cur_input_ids.device, dtype=cur_input_ids.dtype))
                cur_labels_im_replaced.append(torch.full((image_token_len,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                if cur_attention_mask[image_token_indices[i+1]]:    
                    cur_attention_mask_im_replaced.append(torch.full((image_token_len,), 1, device=cur_attention_mask.device, dtype=cur_attention_mask.dtype))
                    cur_position_ids_im_replaced.append(torch.arange(index, index+image_token_len, dtype=torch.long, device=cur_input_ids.device))
                    index += image_token_len
                else:
                    cur_attention_mask_im_replaced.append(torch.full((image_token_len,), 0, device=cur_attention_mask.device, dtype=cur_attention_mask.dtype))
                    cur_position_ids_im_replaced.append(torch.full((image_token_len,), 0, device=cur_input_ids.device, dtype=torch.long))
        
        input_ids_im_replaced.append(torch.cat(cur_input_ids_im_replaced))
        labels_im_replaced.append(torch.cat(cur_labels_im_replaced))
        attention_mask_im_replaced.append(torch.cat(cur_attention_mask_im_replaced))
        position_ids_im_replaced.append(torch.cat(cur_position_ids_im_replaced))
    
    # Truncate sequences to max length as image embeddings can make the sequence longer
    new_input_ids = [x[0:max_length] for x in input_ids_im_replaced]
    new_labels = [x[0:max_length] for x in labels_im_replaced]
    new_attention_mask = [x[0:max_length] for x in attention_mask_im_replaced]
    new_position_ids = [x[0:max_length] for x in position_ids_im_replaced]
    new_input_ids = torch.stack(new_input_ids)
    new_labels = torch.stack(new_labels)
    new_attention_mask = torch.stack(new_attention_mask)
    new_position_ids = torch.stack(new_position_ids)
    return new_input_ids, new_labels, new_attention_mask, new_position_ids


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_token_len: int = 576
    image_position: int = 35

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        image_token_len = self.image_token_len
        image_position = self.image_position

        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        max_length = self.tokenizer.model_max_length

        padding_side = self.tokenizer.padding_side 
        if padding_side == "left":
            input_ids = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (max_length - t.shape[0], 0), 'constant', self.tokenizer.pad_token_id) for t in input_ids]
            labels = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, ( max_length - t.shape[0], 0), 'constant', IGNORE_INDEX) for t in labels]
        else:
            input_ids = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (0, max_length - t.shape[0]), 'constant', self.tokenizer.pad_token_id) for t in input_ids]
            labels = [t[:max_length] if t.shape[0] >= max_length else torch.nn.functional.pad(t, (0, max_length - t.shape[0]), 'constant', IGNORE_INDEX) for t in labels]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask= input_ids.ne(self.tokenizer.pad_token_id)
        # insert dummy image
        for i in range(len(input_ids)):
            if (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == 0:
                cur_input_ids_tmp = input_ids[i].clone()
                cur_input_ids_tmp[image_position+1:] = input_ids[i, image_position:-1]
                cur_input_ids_tmp[image_position] = IMAGE_TOKEN_INDEX
                input_ids[i] = cur_input_ids_tmp

                cur_labels_tmp = labels[i].clone()
                cur_labels_tmp[image_position+1:] = labels[i, image_position:-1]
                cur_labels_tmp[image_position] = IGNORE_INDEX
                labels[i] = cur_labels_tmp

                cur_attention_mask_tmp = attention_mask[i].clone()
                cur_attention_mask_tmp[image_position+1:] = attention_mask[i, image_position:-1]
                cur_attention_mask_tmp[image_position] = False
                attention_mask[i] = cur_attention_mask_tmp
        new_input_ids, new_labels, new_attention_mask, new_position_ids = prepare_multimodal_data(input_ids, labels, attention_mask, image_token_len, max_length)
        batch = dict(
            input_ids=new_input_ids,
            labels=new_labels,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer,
        image_token_len=data_args.image_token_len,
        image_position=data_args.image_position
    )
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def convert_to_bf16_except_llama(model):
    # Loop through all modules and their respective parameters

    # Assuming model is your PyTorch model
    for name, param in model.named_parameters():
        # Check if 'model.layers' is not in the parameter name
        if 'model.layers' not in name:
            # Convert parameter to bfloat16
            param.data = param.data.to(torch.bfloat16)
            # if param.requires_grad:
            #     # Also convert the grad attribute if it exists
            #     param.grad = param.grad.to(torch.bfloat16) if param.grad is not None else None
            print(f"{name} this is converted")

        else:
            print(f"{name} this is not converted")


def patch_llama_reset_parameters():
    from transformers.models.llama.modeling_llama import (
        LlamaRMSNorm,
        LlamaMLP,
        LlamaAttention,
        LlamaDecoderLayer,
        LlamaModel,
        LlamaForCausalLM,
        LlamaForSequenceClassification,
    )

    def llama_rms_norm_reset_parameters(self):
        self.weight.data.fill_(1.0)

    def llama_mlp_reset_parameters(self):
        if self.gate_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.gate_proj.weight)
        if self.up_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.up_proj.weight)
        if self.down_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.down_proj.weight)

    def llama_attention_reset_parameters(self):
        if self.q_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.q_proj.weight)
            if self.config.attention_bias:
                nn.init.constant_(self.q_proj.bias, 0.0)
        if self.k_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.k_proj.weight)
            if self.config.attention_bias:
                nn.init.constant_(self.k_proj.bias, 0.0)
        if self.v_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.v_proj.weight)
            if self.config.attention_bias:
                nn.init.constant_(self.v_proj.bias, 0.0)
        if self.o_proj.weight.dim() > 1:
            nn.init.xavier_uniform_(self.o_proj.weight)
            if self.config.attention_bias:
                nn.init.constant_(self.o_proj.bias, 0.0)

    def llama_decoder_layer_reset_parameters(self):
        self.self_attn.reset_parameters()
        self.mlp.reset_parameters()
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()

    def llama_model_reset_parameters(self):
        self.embed_tokens.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()

    def llama_for_causal_lm_reset_parameters(self):
        self.model.reset_parameters()
        if self.lm_head.weight.dim() > 1:
            self.lm_head.reset_parameters()

    def llama_for_sequence_classification_reset_parameters(self):
        self.model.reset_parameters()
        if self.score.weight.dim() > 1:
            self.score.reset_parameters()


    LlamaRMSNorm.reset_parameters = llama_rms_norm_reset_parameters
    LlamaMLP.reset_parameters = llama_mlp_reset_parameters
    LlamaAttention.reset_parameters = llama_attention_reset_parameters
    LlamaDecoderLayer.reset_parameters = llama_decoder_layer_reset_parameters
    LlamaModel.reset_parameters = llama_model_reset_parameters
    LlamaForCausalLM.reset_parameters = llama_for_causal_lm_reset_parameters
    LlamaForSequenceClassification.reset_parameters = llama_for_sequence_classification_reset_parameters


import torch_xla
import os
XLA_DISABLE_FUNCTIONALIZATION = bool(
    os.environ.get('XLA_DISABLE_FUNCTIONALIZATION', False))


@torch.no_grad()
def _shard_parameters_(self, params_to_shard) -> None:
    """
    At initialization we wrap a module with full parameters and shard the
    parameters in-place. Sharding is implemented by viewing each parameter
    as a 1D Tensor and retaining only a single slice, where the slice size
    is determined by the number of data parallel workers.

    Wrapping modules with many small parameters (or with a very large data
    parallel world size) will result in many small parameter shards and slow
    performance. In this case it's better to set *``flatten_parameters``* to
    ``True``, so that all of the small parameters in the module are combined
    into a single contiguous Tensor and sharded once.

    After this initial sharding is complete, the user can initialize a
    ``torch.optim.Optimizer`` in the usual way, i.e.::

    .. code-block:: python

        optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)

    The optimizer will see only a single slice of parameters and will thus
    allocate less memory for optimizer state, avoiding redundancy across
    data parallel workers.

    Note: this method is implemented in a different manner from
    ``fairscale.nn.FullyShardedDataParallel``. Here we delete the original
    module parameters and create new sharded parameter tensors (instead of
    making sharded tensors an attribute of the original parameters). This
    make it easier to handle things (e.g. freeing parameters) on XLA.
    """

    #print("I actually use this to shard models!")
    if len(params_to_shard) > 0:
      # When freeing the full parameters, we point their internal XLATensor to this placeholder
      # (so that the XLA compiler can reuse the memory storage).
      self._dummy_data_placeholder = torch.zeros(
          1, dtype=self.compute_dtype, device=self.xla_device)

    # get the module names of each full parameter to shard
    params_to_shard_set = set(params_to_shard)
    assert len(params_to_shard_set) == len(params_to_shard), \
        "params_to_shard should not have dups"
    full_param_infos = []
    shared_full_param_memo = {}
    shared_full_param_infos = []
    full_params = []
    for module_name, m in self.named_modules():
      for n, p in m.named_parameters(recurse=False):
        if p.dtype != torch.float32:
          #raise TypeError("only fp32 parameters are supported")
          p.data = p.data.to(torch.float32)
        if p in params_to_shard_set:
          if p in shared_full_param_memo:
            mname, shared_m, shared_n = shared_full_param_memo[p]
            shared_full_param_infos.append(
                (module_name, mname, m, n, shared_m, shared_n))
          else:
            shared_full_param_memo[p] = (module_name, m, n)
            full_param_infos.append((module_name, m, n))
            full_params.append(p)
    assert len(full_params) == len(params_to_shard_set), \
        f"there are parameters in params_to_shard not belonging to this module."
    del shared_full_param_memo
    self.full_params = full_params
    self.full_param_infos = full_param_infos
    self.shared_full_param_infos = shared_full_param_infos

    # allocate and register new sharded parameters
    self.sharded_params = []
    for idx, (module_name, m, n) in enumerate(self.full_param_infos):
        p = self.full_params[idx]
        # print("rank and device are:", self.rank, p.device, "module name is", module_name, p.requires_grad)
        # if self.rank == 0:
        #   # Move the tensor to the XLA device if it's not already on XLA
        #   # if p.device != self.xla_device:
        #   #   p = p.to(self.xla_device)
        #   p = self._broadcast(p, src=0)
        # else:
        #   # Create a placeholder tensor on the XLA device for other ranks
        #   p = torch.empty_like(p, device="cpu", requires_grad=p.requires_grad)
        #   # Receive the broadcasted full parameter from rank 0
        #   p = self._broadcast(p, src=0)
        # print("rank and device are:", self.rank, p.device, "module name is", module_name, p.requires_grad, "finished broadcast")

        assert not hasattr(p, "_is_sharded")

        shard_data = self._get_shard(p)

        if shard_data.device != self.xla_device:
            # cast to XLA device if not already on XLA
            shard_data = shard_data.to(self.xla_device)
        p_shard = nn.Parameter(shard_data, requires_grad=p.requires_grad)
        p_shard._is_sharded = True
        p_shard._orig_size = p.size()
        p_shard._orig_name = f"{module_name}.{n}"
        p_shard._name = f"_fsdp_shard.{p_shard._orig_name}".replace(
            ".", "_FSDP_SHARD_SEPARATOR_")
        self.register_parameter(p_shard._name, p_shard)
        self.sharded_params.append(p_shard)
        if p.device != self.xla_device:
            # cast to XLA device if not already on XLA
            p = p.to(self.xla_device).requires_grad_(p.requires_grad)
            # update p in full_params since id(p) changed after the casting
            self.full_params[idx] = p
        # Free the full parameter storage (here we free its internal XLATensor) but keep the tensor itself
        # for auto-grad tracing (like `torch.autograd.Variable` before the tensor-variable merge).
        if XLA_DISABLE_FUNCTIONALIZATION:
            p.data = p.new_zeros(1)  # Old behavior before Functionalization.
        else:
            torch_xla._XLAC._replace_xla_tensor(p, p.new_zeros(1))
        p._sharded_param = p_shard  # add a handle to the sharded parameter
        p._has_full_param = False
        # deregister the full parameter tensors from their modules (so that they won't
        # appear in the FSDP model's `parameters()` or `named_parameters()` outputs;
        # only the sharded parameters should appear in the FSDP model's `parameters()`)
        assert n in m._parameters
        m._parameters.pop(n)
        object.__setattr__(m, n, p)

    # also deregister the shared parameters
    for _, _, m, n, shared_m, shared_n in self.shared_full_param_infos:
        assert n in m._parameters
        m._parameters.pop(n)
        shared_p = getattr(shared_m, shared_n)
        object.__setattr__(m, n, shared_p)

    assert len(self.sharded_params) == len(self.full_params)

from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel
XlaFullyShardedDataParallel._shard_parameters_ = _shard_parameters_
import re


def train(INDEX, attn_implementation=None):
    # def train(attn_implementation=None):

    global local_rank

    logger.info(f"Training on index {INDEX}. Local rank: {local_rank}")

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    #print(model_args, data_args, training_args)
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    #compute_dtype = torch.float32

    # Forward
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = (self.weight * hidden_states).to(input_dtype)
        return output

    transformers.models.llama.modeling_llama.LlamaRMSNorm.forward = forward
    transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = forward

	#patch_llama_reset_parameters()
    logger.info("I changed forward!")

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        logger.info(f"Loading model in {training_args.bits}bit")
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))
    else:
        logger.info(f"Loading model in full precision")
    use_cohere=False
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            logger.warning(f"MPT model, loading LlavaMptForCausalLM: {model_args.model_name_or_path}")
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            # Assuming model_args.model_name_or_path is a string that includes the model size
            model_name = model_args.model_name_or_path

            # Regular expression to find the number of parameters in the model's name (assuming a convention like 'ModelName-30b')
            match = re.search(r'(\d+)b', model_name)
            num_parameters_billion = float(match.group(1)) if match else 0

            # Determine if bfloat16 should be used based on the model's size
            use_bfloat16 = training_args.bf16 or num_parameters_billion > 30
            if "mixtral" in model_name.lower():
                model = LlavaMixtralForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
                transformers.models.mixtral.modeling_mixtral.MixtralRMSNorm.forward = forward
            elif "c4ai" in model_name.lower():
                model = LlavaCohereForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=training_args.cache_dir,
                    torch_dtype=torch.bfloat16,
                    **bnb_model_from_pretrained_args
                )
                use_cohere=True
            
            elif "mistral" in model_name.lower():
                logger.warning(f"Vision tower, loading LlavaMistralForCausalLM: {model_args.model_name_or_path}")

                # replace training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer
                if (
                    hasattr(training_args, 'fsdp_config') and
                    'transformer_layer_cls_to_wrap' in training_args.fsdp_config.keys()
                ):
                    logger.warning(f"Replacing training_args.fsdp_config.transformer_layer_cls_to_wrap with MistralDecoderLayer. Previous value: {training_args.fsdp_config['transformer_layer_cls_to_wrap']}")
                    training_args.fsdp_config["transformer_layer_cls_to_wrap"] = ["MistralDecoderLayer"]

                model = LlavaMistralForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=training_args.cache_dir,
                    do_sample=True,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
                transformers.models.mistral.modeling_mistral.MistralRMSNorm.forward = forward
            else:
                logger.warning(f"Vision tower, loading LlavaLlamaForCausalLM: {model_args.model_name_or_path}")
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    do_sample=True,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )

            # model = LlavaLlamaForCausalLM.from_pretrained(
            # 		model_args.model_name_or_path,
            # 		cache_dir=training_args.cache_dir,
            # 		torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            # 		**bnb_model_from_pretrained_args
            # 	)
            #from torch_xla.core.xla_model import broadcast_master_param

    else:
        logger.warning(f"No vision tower, loading pure language model: {model_args.model_name_or_path}")
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    model.generation_config.do_sample = True

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    logger.info("Model loaded.")

    # # XLA wait for model to be loaded
    # logger.warning("XLA: Waiting for model to be loaded")
    # import torch_xla.core.xla_model as xm
    # xm.rendezvous('model_loaded')
    # logger.error("BREAKING HERE"); exit()

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        logger.info("Using gradient checkpointing")
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        logger.info("Adding LoRA adapters...")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    logger.info("Configuring tokenizer...")
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast = use_cohere
        )
        print("tokenizer is", tokenizer)
        
    print("tokenizer id before operation is", tokenizer.pad_token_id)


    logger.info(f"Model Conv Version: {model_args.version}")
    logger.info(f"Default conversation version: {conversation_lib.default_conversation.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        print("Model version", model_args.version)
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            logger.warning(f"Conversation version {model_args.version} not found. Using default `vicuna_v1`")
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    logger.info(f"Default conversation version: {conversation_lib.default_conversation.version}")


    if use_cohere:
        tokenizer.pad_token_id = 1
        print("tokenizer id is", tokenizer.pad_token_id)

    if model_args.vision_tower is not None:
        logger.info("Initializing vision modules...")
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        # vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        vision_tower.to(dtype=torch.bfloat16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            logger.info("Tuning multimodal mlp adapter only...")
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            logger.info("Freezing multimodal mlp adapter...")
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            logger.info(f"Initializing vision modules in {training_args.bits}bit")
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
        logger.info("Vision modules initialized.")

    if training_args.bits in [4, 8]:
        logger.info(f"Initializing model in {training_args.bits}bit")
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    logger.info("Configuring data module...")
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)


    ### Implement FSDP


    # import torch_xla.core.xla_model as xm
    # from pprint import pprint
    # from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
    # fsdp_wrap = lambda m: FSDP(m, compute_dtype=torch.bfloat16, shard_param_on_dim_0=True, pin_layout_in_collective_ops=True)
    # import inspect
    # forward_signature = inspect.signature(model.forward.__func__)
    # model = model.to(torch.float32)
    # model = fsdp_wrap(model)
    # model.forward.__func__.__signature__ = forward_signature

    # # Patch `xm.optimizer_step` not to reduce gradients in this case,
    # # as FSDP does not need gradient reduction over sharded parameters.
    # # Note: this ultimately should be something to be implemented in the Hugging Face trainer
    # # to directly call `optimizer.step()` when the model is an FSDP instance,
    # # but we chose to patch it here to get a standalone example without changing the Hugging Face trainer
    # def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
    #     loss = optimizer.step(**optimizer_args)
    #     if barrier:
    #         xm.mark_step()
    #     return loss

    # xm.optimizer_step = patched_optimizer_step


    # convert_to_bf16_except_llama(model)
    if training_args.bf16:
        model = model.to(dtype = torch.float32)

    # gcloud_callback = GCloudRsyncCallback(training_args.output_dir, training_args.gcs_output_dir, training_args.gcp_project)

    logger.info("Configuring trainer...")
    trainer = LLaVATrainer(model=model,
                        tokenizer=tokenizer,
                        args=training_args,
                        # callbacks=[gcloud_callback],
                        **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logger.info(f"Resuming from checkpoint: {training_args.output_dir}")
        trainer.train(resume_from_checkpoint=True)
    else:
        logger.info(f"Starting training: {training_args.output_dir}")
        trainer.train()

    logger.info(f"Training finished: {training_args.output_dir}")

    trainer.save_state()

    model.config.use_cache = True

    logger.info("Saving model...")
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
