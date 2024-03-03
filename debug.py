import torch_xla.core.xla_model as xm

def get_tpu_cores():
    # Returns the number of available TPU cores.
    tpu_cores = xm.xrt_world_size()
    print(f"Number of TPU Cores: {tpu_cores}")

if __name__ == "__main__":
    get_tpu_cores()
