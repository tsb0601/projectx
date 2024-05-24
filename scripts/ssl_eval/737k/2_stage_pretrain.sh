#!/bin/bash
set -e

################# Utils #################

# Function to print logs with timestamp and color the time
log() {
    printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

################# Parse Arguments #################
log "> ssl_eval/737k/2_stage_pretrain.sh $@"

pretrain_data="1.2M"  # default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run_name)
        run_name="$2"
        shift 2
        ;;
    --vision_tower)
        vision_tower="$2"
        shift 2
        ;;
    --image_token_len)
        image_token_len="$2"
        shift 2
        ;;
    --per_device_batch_size)
        per_device_batch_size="$2"
        shift 2
        ;;
    --batch_size)
        batch_size="$2"
        shift 2
        ;;
    --ckpt_path)
        ckpt_path="$2"
        shift 2
        ;;
    --pretrain_data)
        pretrain_data="$2"
        shift 2
        ;;
    *)
        log "Unknown argument: $1"
        exit 1
        ;;
  esac
done

if [ -z "$ckpt_path" ]; then
    ckpt_path=./checkpoints/$run_name/pretrain
    log "Checkpoint path not set. Using default path: $ckpt_path"
fi

# pretrain options:
# 0.5M: llava-1.5
# --data_path ./blip_laion_cc_sbu_558k.jsonl \
# --image_folder /mnt/disks/storage/data/pretrain_data \
# 1.2M: sharegpt4v
# --data_path ./pretrain.jsonl
# --image_folder /mnt/disks/storage/data/finetune_data
# 10M:??? // TBD
# TBD

# validate pretrain choice // set args
if [ "$pretrain_data" == "0.5M" ]; then
    log "Pretraining with 0.5M data ..."
    data_path="./blip_laion_cc_sbu_558k.jsonl"
    image_folder="/mnt/disks/storage/data/pretrain_data/sbu558k/"
elif [ "$pretrain_data" == "1.2M" ]; then
    log "Pretraining with 1.2M data ..."
    data_path="./pretrain.jsonl"
    image_folder="/mnt/disks/storage/data/finetune_data"
else
    log "Invalid pretrain data choice. Current options: 0.5M, 1.2M"
    exit 1
fi


################# Check for previous completed runs #################


gcloud_path="gs://us-central2-storage/cambrian/checkpoints/ssl_exps/2stage_737k/$run_name/pretrain"

# Check if the checkpoint path exists on GCS
log "Checking for previous completed runs at $gcloud_path ..."
if gsutil ls $gcloud_path > /dev/null 2>&1; then
    log "Checkpoint path exists on GCS."
    
    # Capture the output of gsutil ls
    files_output=$(gsutil ls -l $gcloud_path)
    
    # Check for the presence of required files
    required_files=(
        "config.json"
        "trainer_state.json"
    )
    log "Checking for required files: ${required_files[@]}..."
    all_required_files_exist=true
    for file in "${required_files[@]}"; do
        if ! log "$files_output" | grep -q $file; then
            log "$file is missing. Retraining..."
            all_required_files_exist=false
            break
        fi
    done
    if $all_required_files_exist; then
        log "All required files exist."
        
        # Extract the world size from the first shard
        first_shard=$(log "$files_output" | grep -o 'mm_projector_rank-00000000-of-[0-9]*\.pth' | head -n 1)
        if [ -n "$first_shard" ]; then
            world_size=$(log $first_shard | sed -n 's/.*rank-[0-9]*-of-0*\([1-9][0-9]*\).*/\1/p')
            log "World size: $world_size"
            
            # Check if all shards exist
            all_shards_exist=true
            for ((i=0; i<$world_size; i++)); do
                shard_file=$(printf "mm_projector_rank-%08d-of-%08d.pth" $i $world_size)
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
                log "Checkpoint is incomplete. Retraining the model."
            fi
        else
            log "First shard not found. Retraining the model."
        fi
    fi
fi

################# Pretrain  #################

log "Running pretraining for $run_name ..."

python llava/train/train_tpu.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path "$data_path" \
    --image_folder "$image_folder" \
    --vision_tower "$vision_tower" \
    --image_token_len "$image_token_len" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 False \
    --output_dir $ckpt_path \
    --num_train_epochs 1 \
    --per_device_train_batch_size $per_device_batch_size \
    --batch_size $batch_size \
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
    --report_to wandb \
    --run_name $run_name\_pretrain \
    --fsdp "full_shard" \
    --fsdp_config fsdp_config.json \
    --unpad True

# Check if the checkpoint path exists
if [ ! -d "$ckpt_path" ]; then
    log "Checkpoint path does not exist. Exiting..."
    exit 1
fi

log "Training finished for experiment $run_name. Syncing checkpoints to GCS..."

gcloud alpha storage rsync $ckpt_path $gcloud_path

log "Syncing finished for experiment $run_name. Checkpoints are now available at $gcloud_path"
