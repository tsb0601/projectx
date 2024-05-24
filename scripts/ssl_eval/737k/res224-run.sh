#!/bin/bash

export PJRT_DEVICE=TPU
export XLA_USE_BF16=0
export WANDB_ENTITY=nyu-visionx
export WANDB_PROJECT=llava

# take groups command line argument
GROUP=$1  # group in the csv, or "all"

# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set the path to the experiments file relative to the script directory
EXPERIMENTS_FILE="$SCRIPT_DIR/res224-experiments.csv"

# Check if the experiments file exists
if [ ! -f "$EXPERIMENTS_FILE" ]; then
    echo "Experiments file not found: $EXPERIMENTS_FILE"
    exit 1
fi

echo "Reading experiments from $EXPERIMENTS_FILE ..."
echo ""

# Set the path to the success log file relative to the script directory
SUCCESS_LOG="success.log"


# Read the experiments file
while IFS=',' read -r name vision_tower image_token_len ran group; do
    # Skip the header line
    if [[ $name == "name" ]]; then
        continue
    # if not in the passed group or "all" group, skip
    elif [[ $group != $GROUP && $GROUP != "all" ]]; then
        echo "Skipping experiment $name in group '$group' ... Processing group $GROUP"
        continue
    elif [[ $ran == "yes" ]]; then
        echo "Skipping previously ran experiment $name ..."
        continue
    fi



    CKPT_NAME="vicuna-7b-${name}-737k-bs512-res224"

    echo "running experiment $CKPT_NAME ..."

    python llava/train/train_tpu.py \
        --model_name_or_path lmsys/vicuna-7b-v1.5 \
        --version v1 \
        --data_path /mnt/disks/storage/data/finetune_data/jsons/737k.jsonl \
        --image_folder /mnt/disks/storage/data/finetune_data \
        --vision_tower "$vision_tower" \
        --image_token_len "$image_token_len" \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 False \
        --output_dir ./checkpoints/$CKPT_NAME \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100000 \
        --save_total_limit 1 \
        --learning_rate 4e-5 \
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

    CKPT_PATH=./checkpoints/$CKPT_NAME

    # Check if the checkpoint path exists
    if [ ! -d "$CKPT_PATH" ]; then
        echo "Checkpoint path does not exist. Exiting..."
        exit 1
    fi

    echo "Training finished for experiment $name. Syncing checkpoints to GCS..."

    gcloud_path="gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/$CKPT_NAME"
    gcloud alpha storage rsync $CKPT_PATH $gcloud_path

    echo "Syncing finished for experiment $name. Checkpoints are now available at $gcloud_path"

    # Write the successful experiment details to the success log file
    echo "$name,$CKPT_PATH,$(date)" >> "$SUCCESS_LOG"

done < "$EXPERIMENTS_FILE"
