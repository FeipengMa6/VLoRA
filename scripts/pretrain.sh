#!/bin/bash

PYTHONPATH=./ torchrun --nproc_per_node 8 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./playground/data/capsfusion_113m_full_data.json \
    --image_folder ./playground/data/capsfusion \
    --vision_tower clip-vit-large-patch14-336 \
    --mm_projector_type vlora \
    --tune_mm_mlp_adapter True \
    --tune_vision_tower False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/vlora-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 20 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name vlora-7b-pretrain \
    --vlora_dim 1024 \
    --vlora_depth 8 \
    --vlora_visual_dim 1024 \
    --vlora_pos_num 576 \
    --vlora_llm_dim 4096 \
    --vlora_llm_depth 32 \
    --vlora_rank 64 \
    --vlora_alpha 64 \
    --vlora_type qkvom \
    --weights_sep True \
    --skip_layers 4
