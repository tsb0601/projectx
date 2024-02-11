from llava.train.train import train

#from llava.train.train_spmd import train

if __name__ == "__main__":
    #train()
    import multiprocessing as mp
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


    # import safetensors
    # import torch_xla
    # from safetensors.torch import storage_size

    # # Using Patch to fix the bug
    # safetensors.torch.storage_ptr = lambda tensor: torch_xla._XLAC._xla_get_tensor_id(tensor)

    # def id_tensor_storage(tensor):
    #     """
    #     Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    #     example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    #     guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    #     non-overlapping lifetimes may have the same id.
    #     """

    #     unique_id = torch_xla._XLAC._xla_get_tensor_id(tensor)
      

    #     return tensor.device, unique_id, storage_size(tensor)
    
    mp.set_start_method('spawn', force=True)
    xmp.spawn(train, args=(None,))

