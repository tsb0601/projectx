import torch_xla

# Using Patch to fix the bug
#safetensors.torch.storage_ptr = lambda tensor: 0


# def id_tensor_storage(tensor):
#     """
#     Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
#     example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
#     guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
#     non-overlapping lifetimes may have the same id.
#     """
#     if "xla" in tensor.device.type:
#         # NOTE: xla tensors dont have storage
#         # use some other unique id to distinguish.
#         # this is a XLA tensor, it must be created using torch_xla's
#         # device. So the following import is safe:
#         import torch_xla
#         unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
#         print("I used this function!!")
#     else:
#         unique_id = storage_ptr(tensor)

#     return tensor.device, unique_id, storage_size(tensor)

# import transformers
# transformers.pytorch_utils.id_tensor_storage = id_tensor_storage

#from llava.train.train import train
from llava.train.train_fsdp import train

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import types

# SPIECE_UNDERLINE = "▁"
# def tokenize(self, text: "TextInput", add_special_tokens=False, **kwargs) -> List[str]:
#     """
#     Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
#     first token is special.
#     """
#     if self.legacy or len(text) == 0:
#         return super(LlamaTokenizer, self).tokenize(text, **kwargs)

#     text = text.replace(SPIECE_UNDERLINE, " ")
#     if self.add_prefix_space:
#         text = SPIECE_UNDERLINE + text

#     tokens = super(LlamaTokenizer, self).tokenize(text, **kwargs)

#     if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
#         tokens = tokens[1:]
#     return tokens
# LlamaTokenizer.tokenize = tokenize

# import wandb
# wandb.login(key='618eb3b78242f01000855a123d29e2ac98a60f30')

if __name__ == "__main__":
    #train()
    import multiprocessing as mp
    import torch_xla.distributed.xla_multiprocessing as xmp
   
   
    mp.set_start_method('spawn', force=True)
    xmp.spawn(train, args=(None,))

