## TPU Pod
The TPU runtime-version is tpu-ubuntu2204-base

## Running the Code

Note that for both pretraining and fine-tuning, I am using a global batch size of 256.

### Pretraining

```
gcloud compute tpus tpu-vm ssh your_tpu_name --zone=us-central2-b --worker=all --command="
cd llava_base &&
pip install --upgrade pip setuptools &&
pip install -e . &&
pip install wandb &&
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html &&
export PJRT_DEVICE=TPU &&
export XLA_USE_BF16=0 &&
sh pretrain_fsdp.sh"
```

### Finetuning

```
gcloud compute tpus tpu-vm ssh your_tpu_name --zone=us-central2-b --worker=all --command="
cd llava_base &&
pip install --upgrade pip setuptools &&
pip install -e . &&
pip install wandb &&
pip install torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html &&
export PJRT_DEVICE=TPU &&
export XLA_USE_BF16=0 &&
sh finetune_fsdp.sh"
```
