#!/bin/bash

export PJRT_DEVICE=TPU
export XLA_USE_BF16=0
export WANDB_ENTITY=nyu-visionx
export WANDB_PROJECT=llava
export CKPT_NAME="TPU-llava-v1.5-8b-finetune-6993k-lr6e-5"

python llava/train/train_tpu.py \
    --model_name_or_path /mnt/disks/storage/llm_ckpts/Meta-Llama-3-8B-Instruct \
    --version llama_v3 \
    --data_path /mnt/disks/storage/data/finetune_data/6993kL.jsonl \
    --image_folder /mnt/disks/storage/data/finetune_data \
    --vision_tower siglip/CLIP-ViT-SO400M-14-384 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir ./checkpoints/$CKPT_NAME \
    --num_train_epochs 1 \
    --image_token_len 729 \
    --image_position 92 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 6e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $CKPT_NAME \
    --fsdp "full_shard" \
    --fsdp_config fsdp_config.json

CKPT_PATH=checkpoints/$CKPT_NAME

# check if the checkpoint path exists
if [ ! -d "$CKPT_PATH" ]; then
    echo "Checkpoint path does not exist. Exiting..."
    exit 1
fi

echo "Training finished. Syncing checkpoints to GCS..."

gcloud alpha storage rsync $CKPT_PATH gs://us-central2-storage/cambrian/checkpoints/$CKPT_NAME

echo "Syncing finished. Checkpoints are now available at gs://us-central2-storage/cambrian/checkpoints/$CKPT_NAME"