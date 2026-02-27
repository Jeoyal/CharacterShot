#!/usr/bin/env bash
# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
# Model Configuration
MODEL_ARGS=(
    --model_path "zai-org/CogVideoX-5b-I2V"
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "sft"
    # --pose_model_path ckpt from 2dpretrain stage
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "./result/CharacterShot/4dfinetune/"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "data/i2v/CharacterShot"
    --video_column "encoded_videos_multiview.txt"
    --pose_column "poses_vae_multiview.txt"
    --image_column "image_raw_multiview.txt"
    --train_resolution "25x480x720"  # (frames x height x width), frames should be 8N+1 and height, width should be multiples of 16
    --func_type "4dfinetune"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 2 # number of training epochs
    --seed 42 # random seed
    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 2
    --learning_rate 5e-5
    --gradient_accumulation_steps 2
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
    --checkpointing_steps 500 # save checkpoint every x steps
    --checkpointing_limit 6 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint if you want to resume from a checkpoint, otherwise, comment this line
)

accelerate launch \
  --config_file accelerate_config_16gpus.yaml \
  --machine_rank ${RANK} \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  train.py \
  "${MODEL_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}" \
  "${DATA_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  "${SYSTEM_ARGS[@]}" \
  "${CHECKPOINT_ARGS[@]}" \
  "${VALIDATION_ARGS[@]}"
