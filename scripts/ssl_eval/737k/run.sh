#!/bin/bash
set -e

################# Utils #################

# Function to print logs with timestamp and color the time
log() {
    printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

################# Parse Arguments #################

export PJRT_DEVICE=TPU
export XLA_USE_BF16=0
export WANDB_ENTITY=nyu-visionx
export WANDB_PROJECT=llava

# take groups command line argument
GROUP=$1  # group in the csv, or "all"
POD_SIZE=$2  # v4-128, v4-256, v4-512

# check that the pod size is set
if [[ -z "$POD_SIZE" ]]; then
    log "Pod size not set. Exiting..."
    exit 1
fi

BATCH_SIZE=512
# calculate per device batch size using the pod size
PER_DEVICE_BATCH_SIZE=$((BATCH_SIZE / POD_SIZE * 2))

log "Using per device batch size $PER_DEVICE_BATCH_SIZE with pod size $POD_SIZE to get total batch size $BATCH_SIZE"


# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set the path to the experiments file relative to the script directory
EXPERIMENTS_FILE="$SCRIPT_DIR/experiments.csv"

# Check if the experiments file exists
if [ ! -f "$EXPERIMENTS_FILE" ]; then
    log "Experiments file not found: $EXPERIMENTS_FILE"
    exit 1
fi

log "Reading experiments from $EXPERIMENTS_FILE ..."
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
        log "Skipping experiment $name in group '$group' ... Processing group $GROUP"
        continue
    elif [[ $ran == "yes" ]]; then
        log "Skipping previously ran experiment $name ..."
        continue
    fi

    run_name="vicuna-7b-${name}-737k-bs512"

    ################# Check for previous completed runs #################

    gcloud_path="gs://us-central2-storage/cambrian/checkpoints/ssl_exps/737k/$run_name"

    # Check if the checkpoint path exists on GCS
    log "Checking for previous completed runs at $gcloud_path ..."
    if gsutil ls $gcloud_path > /dev/null 2>&1; then
        log "Checkpoint path already exists on GCS."
        # Capture the output of gsutil ls
        files_output=$(gsutil ls -l $gcloud_path)

        # Check for the presence of required files
        required_files=(
            "config.json"
            "special_tokens_map.json"
            "tokenizer.model"
            "tokenizer_config.json"
            "trainer_state.json"
            "training_args.bin"
        )
        log "Checking for required files: ${required_files[@]}..."
        all_required_files_exist=true
        for file in "${required_files[@]}"; do
            if ! log "$files_output" | grep -q "$file"; then
                log "$file not found. Retraining..."
                all_required_files_exist=false
                break
            fi
        done
        if $all_required_files_exist; then
            log "All required files exist. Checking for checkpoint shards..."

            # Extract the world size from the first shard
            first_shard=$(log "$files_output" | grep -o 'model_ckpt_rank-00000000*-of-[0-9]*\.pth' | head -n 1)
            if [ -n "$first_shard" ]; then
                world_size=$(log $first_shard | sed -n 's/.*rank-[0-9]*-of-0*\([1-9][0-9]*\).*/\1/p')
                log "World size: $world_size"

                # Check if all shards exist
                all_shards_exist=true
                for ((i=0; i<$world_size; i++)); do
                    shard_file=$(printf "model_ckpt_rank-%08d-of-%08d" $i $world_size)
                    if ! log "$files_output" | grep -q $shard_file; then
                        log "Shard file $shard_file is missing."
                        all_shards_exist=false
                        break
                    fi
                done
                if $all_shards_exist; then
                    log "All $world_size checkpoint shards exist. Skipping previously completed experiment $run_name."
                    exit 0
                else
                    log "Checkpoint is incomplete. Continuing..."
                fi
            else
                log "First shard not found. Continuing..."
            fi
        fi
    fi

    ################# Finetuning #################

    log "running finetuning for $run_name ..."

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
        --output_dir ./checkpoints/$run_name \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
        --batch_size $BATCH_SIZE \
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
        --run_name $run_name \
        --fsdp "full_shard" \
        --fsdp_config fsdp_config.json

    CKPT_PATH=./checkpoints/$run_name

    # Check if the checkpoint path exists
    if [ ! -d "$CKPT_PATH" ]; then
        log "Checkpoint path does not exist. Exiting..."
        exit 1
    fi

    log "Training finished for experiment $name. Syncing checkpoints to GCS..."

    gcloud alpha storage rsync $CKPT_PATH $gcloud_path

    log "Syncing finished for experiment $name. Checkpoints are now available at $gcloud_path"

    # Write the successful experiment details to the success log file
    echo "$name,$CKPT_PATH,$(date)" >> "$SUCCESS_LOG"

done < "$EXPERIMENTS_FILE"
