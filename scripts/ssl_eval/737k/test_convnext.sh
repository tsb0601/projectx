vision_tower=clip-convnext-res1024-interp576
image_token_len=576

run_name="clip-convnext-res1024-interp576-unfreeze-test"
ckpt_path=./checkpoints/$run_name

mm_vision_tower_lr=1e-5
unfreeze_mm_vision_tower=True
BATCH_SIZE=512
POD_SIZE=128
per_device_batch_size=$((BATCH_SIZE / POD_SIZE * 2))
finetune_data_path="/mnt/disks/storage/data/finetune_data/jsons/737k.jsonl"

train_args=(
    --model_name_or_path lmsys/vicuna-7b-v1.5
    --version v1
    --data_path $finetune_data_path
    --image_folder /mnt/disks/storage/data/finetune_data
    --vision_tower "$vision_tower"
    --image_token_len "$image_token_len"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio pad
    --group_by_modality_length True
    --bf16 False
    --output_dir $ckpt_path
    --num_train_epochs 1
    --per_device_train_batch_size $per_device_batch_size
    --batch_size $BATCH_SIZE
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps 1
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps 100000
    --save_total_limit 1
    --learning_rate 4e-5
    --weight_decay 0.
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --tf32 False
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to wandb
    --run_name $run_name\_finetune
    --fsdp "full_shard"
    --fsdp_config fsdp_config.json
    --unpad True
    --unfreeze_mm_vision_tower
    --mm_vision_tower_lr $mm_vision_tower_lr
)

python llava/train/train_tpu.py "${train_args[@]}"
