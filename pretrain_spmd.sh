#!/bin/bash

python llava/train/train_spmd.py \
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
<<<<<<< HEAD
    --per_device_train_batch_size 90 \
=======
<<<<<<< HEAD
    --per_device_train_batch_size 90 \
=======
    --per_device_train_batch_size 256 \
>>>>>>> 367dc5a6665b2a568fb144a856fac396e4280326
>>>>>>> 8792dfb74613dcb89e6f6b9bf4b9eb5d4a29fbad
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
