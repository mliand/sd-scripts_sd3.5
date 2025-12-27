#!/bin/bash
# Example training script for SD3.5 with Gated Attention

# Basic training with headwise gating (recommended, fewer parameters)
python sd3_train_gated.py \
    --pretrained_model_name_or_path "path/to/sd3.5_medium.safetensors" \
    --clip_l "path/to/clip_l.safetensors" \
    --clip_g "path/to/clip_g.safetensors" \
    --t5xxl "path/to/t5xxl.safetensors" \
    --train_data_dir "path/to/training_data" \
    --output_dir "./output" \
    --output_name "sd35_gated" \
    --resolution 1024 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_train_steps 10000 \
    --mixed_precision bf16 \
    --save_every_n_steps 1000 \
    --gate_type headwise \
    --log_gate_stats \
    --log_gate_stats_interval 100 \
    --cache_latents \
    --cache_text_encoder_outputs \
    --gradient_checkpointing

# Training with elementwise gating (more fine-grained control, more parameters)
# python sd3_train_gated.py \
#     --pretrained_model_name_or_path "path/to/sd3.5_medium.safetensors" \
#     --clip_l "path/to/clip_l.safetensors" \
#     --clip_g "path/to/clip_g.safetensors" \
#     --t5xxl "path/to/t5xxl.safetensors" \
#     --train_data_dir "path/to/training_data" \
#     --output_dir "./output" \
#     --output_name "sd35_gated_elementwise" \
#     --resolution 1024 \
#     --train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --learning_rate 1e-5 \
#     --max_train_steps 10000 \
#     --mixed_precision bf16 \
#     --save_every_n_steps 1000 \
#     --gate_type elementwise \
#     --log_gate_stats \
#     --log_gate_stats_interval 100 \
#     --log_gate_stats_detailed \
#     --cache_latents \
#     --cache_text_encoder_outputs \
#     --gradient_checkpointing
