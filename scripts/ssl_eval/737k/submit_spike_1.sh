#!/bin/bash
set -e

POD_SIZE=$1  # v4-128, v4-256, v4-512

# we will submit the following jobs:
# 1. 2-stage 0.5
# 2. 2-stage 
# 3. 2-stage unfrozen


# Get the directory of the current script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


# Function to print logs with timestamp and color the time
log() {
    # printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
    # make this one print red
    printf "\033[31m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

# GROUP="DFN"
run_groups=("siglip-vit-l" "todo")

for run_group in "${run_groups[@]}"
do
    log "[${run_group}] 0. 2-stage 0M"
    cmd="bash $SCRIPT_DIR/run.sh $run_group $POD_SIZE"
    log "> $cmd"
    eval $cmd

    log "[${run_group}] 1. 2-stage 0.5M"
    cmd="bash $SCRIPT_DIR/2_stage.sh --group $run_group --pod_size $POD_SIZE --pretrain_data 0.5M"
    log "> $cmd"
    eval $cmd

    log "[${run_group}] 2. 2-stage"
    cmd="bash $SCRIPT_DIR/2_stage.sh --group $run_group --pod_size $POD_SIZE"
    log "> $cmd"
    eval $cmd

    log "[${run_group}] 3. 2-stage unfrozen"
    cmd="bash $SCRIPT_DIR/2_stage.sh --group $run_group --pod_size $POD_SIZE --unfreeze_mm_vision_tower"
    log "> $cmd"
    eval $cmd
done

log "DONE"
