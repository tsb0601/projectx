# Project Cambrian
![Cambrian Image](cambrian.png)
## Overview

This README provides essential information for setting up and running the LLaVA model training on TPU devices, including both TPU nodes and TPU Pods. Our codebase is designed for stability on TPU nodes with ongoing development for TPU Pod scalability. For detailed LLaVA-related information, please refer to `LLaVA.md`.

## TPU Setup

### Creating TPU Pods

To create a TPU Pod, use the following gcloud commands:

1. Set local environment variables for your TPU Pod:
    ```bash
    TPU_NAME=your_tpu_name
    TPU_TYPE=yourv4type
    PD_NAME=your_pd_name
    ```

2. Create the TPU Pod:
    ```bash
    gcloud alpha compute tpus queued-resources create $TPU_NAME \
        --node-id $TPU_NAME \
        --project nyu-vision-lab \
        --zone us-central2-b \
        --accelerator-type $TPU_TYPE \
        --runtime-version tpu-vm-v4-pt-2.0 \
        --best-effort
    ```
3. Once the TPU Pod is created, ssh into the TPU
    ```bash
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=us-central2-b --worker=all --project "nyu-vision-lab"
    ```
    and run:
    ```bash
    gcloud compute config-ssh
    ```
    Then exit the TPU.

4. Attach a persistent disk to the TPU Pod (see [Attaching Persistent Disks](#attaching-persistent-disks)).
    ```bash
    gcloud alpha compute tpus tpu-vm attach-disk $TPU_NAME \
        --zone=us-central2-b \
        --disk=$PD_NAME \
        --mode=read-only
    ```

5. Setup the TPU Pod (see [Pretraining](#pretraining)).<br>
    On the pod, download the codebase and run the infra setup scripts:
    ```bash
    git clone https://github.com/tsb0601/projectx.git ~/projectx &&
    bash ~/projectx/scripts/infra/mount_disk.bash &&
    bash ~/projectx/scripts/infra/add_env_vars.sh &&
    source ~/.bashrc &&
    bash ~/projectx/scripts/infra/pip_install.sh
    ```

### Attaching Persistent Disks

Datasets and LLM checkpoints are stored on `peter-pd-mllm`. To attach this disk in read-only mode, execute:
```
gcloud alpha compute tpus tpu-vm attach-disk your_tpu_name \
  --zone=us-central2-b \
  --disk=peter-pd-mllm \
  --mode=read-only
``` 


## Running the Code

### Pretraining

The LLaVA model pretraining involves a 2-stage process. Use the following command template for the initial pretraining phase:
```
gcloud compute tpus tpu-vm ssh your_tpu_name --zone=us-central2-b --worker=all --command="
git clone https://github.com/tsb0601/projectx.git ~/projectx &&
bash ~/projectx/scripts/infra/mount_disk.bash &&
bash ~/projectx/scripts/infra/add_env_vars.sh &&
bash ~/projectx/scripts/infra/pip_install.sh &&
cd ~/projectx/ &&
export LD_LIBRARY_PATH=/mnt/disks/storage/envs/anaconda3/lib:$LD_LIBRARY_PATH &&
python llava/train/train_tpu.py \
    --deepspeed /home/tsb/projectx/scripts/zero2.json \
    --model_name_or_path /mnt/disks/storage/llm_ckpts/vicuna1.5 \
    --version plain \
    --data_path /mnt/disks/storage/data/pretrain_data/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/disks/storage/data/pretrain_data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type 'cosine' \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb"
```

Note: `per_device_train_batch_size` in SPMD mode refers to the total batch size, a TorchXLA naming mistake.

## TODOs

### Infrastructure

- Explore and implement SPMD training, crucial for finetuning the entire LLM.
- Address and solve the issue of DDP training interruptions likely caused by TPU bugs to reproduce LLaVA results.

## Others
If you have any questions, please reach out to Peter on Slack at any time.
