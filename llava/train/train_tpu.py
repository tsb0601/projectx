from llava.train.train import train

#from llava.train.train_spmd import train

if __name__ == "__main__":
    #train()
    import multiprocessing as mp
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    
    mp.set_start_method('spawn', force=True)
    xmp.spawn(train, args=(None,))

