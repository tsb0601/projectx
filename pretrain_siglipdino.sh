#!/bin/bash
export PJRT_DEVICE=TPU
export XLA_USE_BF16=0
export WANDB_ENTITY=nyu-visionx
export WANDB_PROJECT=llava
export CKPT_NAME="pretrain-siglip-dino-unpadIII"


python llava/train/train_tpu.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /mnt/disks/storage/data/finetune_data/pretrain.jsonl \
    --image_folder /mnt/disks/storage/data/finetune_data \
    --vision_tower hybridmodel-siglip/CLIP-ViT-SO400M-14-384-and-facebook/dinov2-giant-res378 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --output_dir ./checkpoints/$CKPT_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --unpad True \
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
