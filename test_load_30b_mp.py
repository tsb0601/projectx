import transformers
from ezcolorlog import log_stdout, root_logger as logger


def load(index, model_name):
    logger.info(f"[IDX {index}] transformers.LlamaForCausalLM")

    # logger.info(f"[IDX {index}] Loading LLM: {model_name}")
    # model = transformers.LlamaForCausalLM.from_pretrained(
    #     model_name,
    #     attn_implementation=None,
    #     torch_dtype=None,
    # )
    # print(f"Loaded model {model_name}")

    """flock example -- https://github.com/pytorch/xla/issues/1262#issuecomment-548318533
    lock_file = "tpu.lock"
    fd = open(lock_file, "w")
    fcntl.lockf(fd, fcntl.LOCK_EX)
    #load a model here
    model.train().to(device)
    gc.collect()
    fcntl.lockf(fd, fcntl.LOCK_UN)
    #continue to training
    """

    import fcntl
    import gc
    import torch_xla.core.xla_model as xm

    lock_file = "tpu.lock"
    with open(lock_file, "w") as fd:
        fcntl.lockf(fd, fcntl.LOCK_EX)

        logger.info(f"Taking lock and loading model")

        # load a model here
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_name,
            attn_implementation=None,
            torch_dtype=None,
        )
        model.train().to(xm.xla_device())
        gc.collect()
        logger.info(f"Loaded model {model_name}")

        fcntl.lockf(fd, fcntl.LOCK_UN)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--num_processes", type=int, default=None)
    args = parser.parse_args()

    # load(args.model_name)

    import multiprocessing as mp
    import torch_xla.distributed.xla_multiprocessing as xmp

    logger.info("Starting XMP spawn")
    mp.set_start_method('spawn', force=True)
    xmp.spawn(load, args=(args.model_name,), nprocs=args.num_processes)
    logger.info("Finished XMP spawn")

    # # fork
    # logger.info("Starting XMP fork")
    # mp.set_start_method('fork', force=True)
    # xmp.spawn(load, args=(args.model_name,), start_method='fork')

    # logger.info("Finished XMP fork")