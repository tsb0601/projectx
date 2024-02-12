import safetensors
import torch_xla
from safetensors.torch import storage_size, storage_ptr

# Using Patch to fix the bug
#safetensors.torch.storage_ptr = lambda tensor: 0
import torch_xla.core.xla_model as xm

is_master = xm.get_ordinal() == 0

    # Initialize WandB only in the master process
wandb.login(key='ed3fdff5ab6fba82056002ff9eafa951bf24ec14')

if is_master:
    import wandb
    wandb.init(project='llava_tpu', entity='benchmark_vllm')
else:
    wandb.init(mode="disabled")


def id_tensor_storage(tensor):
    """
    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    """
    if "xla" in tensor.device.type:
        # NOTE: xla tensors dont have storage
        # use some other unique id to distinguish.
        # this is a XLA tensor, it must be created using torch_xla's
        # device. So the following import is safe:
        import torch_xla
        unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
        print("I used this function!!")
    else:
        unique_id = storage_ptr(tensor)

    return tensor.device, unique_id, storage_size(tensor)

import transformers
transformers.pytorch_utils.id_tensor_storage = id_tensor_storage

from llava.train.train import train
#from llava.train.train_fsdp import train

if __name__ == "__main__":
    #train()
    import multiprocessing as mp
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    
   
    mp.set_start_method('spawn', force=True)
    xmp.spawn(train, args=(None,))

