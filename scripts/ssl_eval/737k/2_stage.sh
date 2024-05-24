#!/bin/bash
set -e

export PJRT_DEVICE=TPU
export XLA_USE_BF16=0
export WANDB_ENTITY=nyu-visionx
export WANDB_PROJECT=llava

################# Utils #################

# Function to print logs with timestamp and color the time
log() {
    printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

################# Parse Arguments #################

GROUP=all  # group in the csv, or "all"
# POD_SIZE=$2  # v4-128, v4-256, v4-512
PRETRAIN_DATA="1.2M"  # default
FINETUNE_DATA="737k"  # default
mm_vision_tower_lr=1e-5 # default
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
    --finetune_data)
        FINETUNE_DATA="$2"
        shift 2
        ;;
    --mm_vision_tower_lr)
        mm_vision_tower_lr="$2"
        shift 2
        ;;
    --unfreeze_mm_vision_tower)
        unfreeze_mm_vision_tower="True"
        shift 1
        ;;
    *)
        log "Unknown argument: $1"
        exit 1
        ;;
  esac
done


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


TYPE="2stage"  # default (1.2M pretrain), works with old runs

# if pretrain with 0.5M data, append 0.5M to the type
if [ "$PRETRAIN_DATA" == "0.5M" ]; then
    log "Pretraining with 0.5M data ..."
    TYPE="${TYPE}_pt0.5M"
    gcloud storage cp -n gs://us-central2-storage/cambrian/data/blip_laion_cc_sbu_558k.jsonl ./blip_laion_cc_sbu_558k.jsonl
elif [ "$PRETRAIN_DATA" == "1.2M" ]; then
    log "Pretraining with 1.2M data ..."
    gcloud storage cp -n gs://us-central2-storage/cambrian/data/pretrain.jsonl ./pretrain.jsonl
else
    log "Unknown pretrain data: $PRETRAIN_DATA"
    exit 1
fi

experiment_csv_fn=$TYPE
if [ "$unfreeze_mm_vision_tower" == "True" ]; then
    experiment_csv_fn="unfreeze_${experiment_csv_fn}"
fi


if [ "$FINETUNE_DATA" == "737k" ]; then
    log "Finetuning with 737k data ..."
    finetune_data_path="/mnt/disks/storage/data/finetune_data/jsons/737k.jsonl"
elif [ "$FINETUNE_DATA" == "5186k" ]; then
    log "Finetuning with 5186k data ..."
    finetune_data_path="/mnt/disks/storage/data/finetune_data/5186kL.jsonl"
    experiment_csv_fn+="_ft5186k"
else
    log "Unknown finetune data: $FINETUNE_DATA"
    exit 1
fi

# Set the path to the experiments file relative to the script directory
EXPERIMENTS_FILE="$SCRIPT_DIR/${experiment_csv_fn}_experiments.csv"
# Set the path to the success log file relative to the script directory
SUCCESS_LOG="${TYPE}_success.log"

# Check if the experiments file exists
if [ ! -f "$EXPERIMENTS_FILE" ]; then
    log "Experiments file not found: $EXPERIMENTS_FILE"
    exit 1
fi

log "Reading experiments from $EXPERIMENTS_FILE ..."
log ""


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

    # always use "737k" name for pretrain... poor naming prior
    run_name="${TYPE}-ft-vicuna-7b-${name}-737k-bs${BATCH_SIZE}"
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
        --ckpt_path "$pretrain_ckpt_path" \
        --pretrain_data "$PRETRAIN_DATA"

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
    run_name="${TYPE}-ft-vicuna-7b-${name}-${FINETUNE_DATA}-bs${BATCH_SIZE}"
    CKPT_PATH=./checkpoints/$run_name

    args=(
        --vision_tower "$vision_tower"
        --image_token_len "$image_token_len"
        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE"
        --batch_size "$BATCH_SIZE"
        --projector_path "$projector_path"
        --ckpt_path "$CKPT_PATH"
        --finetune_data_path "$finetune_data_path"
    )
    if [ "$unfreeze_mm_vision_tower" == "True" ]; then
        run_name+="-mmlr$mm_vision_tower_lr-unfreeze"
        args+=(
            --unfreeze_mm_vision_tower "True"
            --mm_vision_tower_lr "$mm_vision_tower_lr"
        )
    fi
    args+=(--run_name "$run_name")

    bash $SCRIPT_DIR/2_stage_finetune.sh "${args[@]}"
    
    wait
    
    ################# Write to Success Log #################

    # Write the successful experiment details to the success log file
    log "$name,$CKPT_PATH,$(date)" >> "$SUCCESS_LOG"

done < "$EXPERIMENTS_FILE"
 