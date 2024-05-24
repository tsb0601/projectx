#!/bin/bash
set -e

export PJRT_DEVICE=TPU
export XLA_USE_BF16=0
export WANDB_ENTITY=nyu-visionx
export WANDB_PROJECT=llava

################# Parse Arguments #################

GROUP=all  # group in the csv, or "all"
# POD_SIZE=$2  # v4-128, v4-256, v4-512
PRETRAIN_DATA="1.2M"  # default
mm_vision_tower_lr=1e-5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
        GROUP="$2"
        shift 2
        ;;
    --pod_size)
        POD_SIZE="$2"
        shift 2
        ;;
    --pretrain_data)
        PRETRAIN_DATA="$2"
        shift 2
        ;;
    --mm_vision_tower_lr)
        mm_vision_tower_lr="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done


# check that the pod size is set
if [[ -z "$POD_SIZE" ]]; then
    echo "Pod size not set. Exiting..."
    exit 1
fi

BATCH_SIZE=512
# calculate per device batch size using the pod size
PER_DEVICE_BATCH_SIZE=$((BATCH_SIZE / POD_SIZE * 2))

echo "Using per device batch size $PER_DEVICE_BATCH_SIZE with pod size $POD_SIZE to get total batch size $BATCH_SIZE"


# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Set the path to the experiments file relative to the script directory
EXPERIMENTS_FILE="$SCRIPT_DIR/unfreeze_2stage_experiments.csv"

# Check if the experiments file exists
if [ ! -f "$EXPERIMENTS_FILE" ]; then
    echo "Experiments file not found: $EXPERIMENTS_FILE"
    exit 1
fi

echo "Reading experiments from $EXPERIMENTS_FILE ..."
echo ""

# Set the path to the success log file relative to the script directory
SUCCESS_LOG="unfreeze_2_stage_success.log"


# donwload pretrain jsonl from gcp
# gcloud storage cp /mnt/disks/storage/data/finetune_data/pretrain.jsonl gs://us-central2-storage/cambrian/data/pretrain.jsonl

gcloud storage cp gs://us-central2-storage/cambrian/data/pretrain.jsonl ./pretrain.jsonl

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

    # reuse the same pretrain path for freeze/unfreeze
    run_name="2stage-ft-vicuna-7b-${name}-737k-bs512"
    CKPT_PATH=./checkpoints/$run_name
    pretrain_ckpt_path="$CKPT_PATH/pretrain"
    projector_path="$pretrain_ckpt_path/mm_projector.bin"

    ################# Pretrain  #################

    bash $SCRIPT_DIR/2_stage_pretrain.sh \
        --run_name "$run_name" \
        --vision_tower "$vision_tower" \
        --image_token_len "$image_token_len" \
        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --ckpt_path "$pretrain_ckpt_path"

    wait

    ################# Consolidate the checkpoints to worker 0 #################

    pt_gcloud_path="gs://us-central2-storage/cambrian/checkpoints/ssl_exps/2stage_737k/$run_name/pretrain"
    # sync both ways to make sure all files are copied
    gcloud alpha storage rsync $pt_gcloud_path $pretrain_ckpt_path
    wait
    sleep 2.5
    gcloud alpha storage rsync $pretrain_ckpt_path $pt_gcloud_path
    wait
    sleep 2.5
    gcloud alpha storage rsync $pt_gcloud_path $pretrain_ckpt_path
    wait
    sleep 2.5
    gcloud alpha storage rsync $pretrain_ckpt_path $pt_gcloud_path
    wait
    sleep 2.5
    gcloud alpha storage rsync $pt_gcloud_path $pretrain_ckpt_path
    wait


    python llava/model/consolidate.py \
        --ckpt_path $pretrain_ckpt_path \
        --ckpt_prefix "mm_projector" \
        --save_filename "mm_projector.bin" \
        --skip_existing
    
    wait

    ################# Finetune  #################

    bash $SCRIPT_DIR/2_stage_finetune.sh \
        --run_name "$run_name-mmlr$mm_vision_tower_lr-unfreeze" \
        --vision_tower "$vision_tower" \
        --image_token_len "$image_token_len" \
        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
        --batch_size "$BATCH_SIZE" \
        --projector_path "$projector_path" \
        --ckpt_path "$CKPT_PATH-unfreeze" \
        --unfreeze_mm_vision_tower True \
        --mm_vision_tower_lr $mm_vision_tower_lr
    
    wait
    
    ################# Write to Success Log #################

    # Write the successful experiment details to the success log file
    echo "$name,$CKPT_PATH,$(date)" >> "$SUCCESS_LOG"

done < "$EXPERIMENTS_FILE"
 