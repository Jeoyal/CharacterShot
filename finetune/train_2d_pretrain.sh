#!/usr/bin/env bash
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "zai-org/CogVideoX-5b-I2V"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "./result/CharacterShot/2dpretrain/"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "data/i2v/CharacterShot"
    --video_column "encoded_videos.txt"
    --pose_column "poses_vae.txt"
    --image_column "image_raw.txt"
    --train_resolution "25x480x720"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
    --func_type "2dpretrain"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 2 # number of training epochs
    --seed 42 # random seed
    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 4
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 1000 # save checkpoint every x steps
    --checkpointing_limit 3 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint if you want to resume from a checkpoint, otherwise, comment this line
)

# Combine all arguments and launch training
accelerate launch --config_file accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
