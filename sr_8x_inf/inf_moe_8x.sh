#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

CONFIG_PATH="configs/extreme_sr/config_sr_FFNMoe_ewa_time_stage.yaml"
CKPT_PATH="models/Unet/sr_8x/moe_stage0.ckpt models/Unet/sr_8x/moe_stage1.ckpt models/Unet/sr_8x/moe_stage2.ckpt models/Unet/sr_8x/moe_stage3.ckpt"
OUTPUT_PATH="output/"
INPUT_PATH="examples/lr"

num_length=4
num_node=0
sample_steps=200

decoder_config="configs/extreme_sr/vae_cfw_aff_unet_8x.yaml"
decoder_ckpt="models/first_stage_models/vq-f4/fa_vae.pth"

export HOST_NUM=1
export INDEX=0
export CHIEF_IP=localhost
export HOST_GPU_NUM=1
port=1234

NUM_PROC=$((HOST_NUM*HOST_GPU_NUM))
echo ${NUM_PROC}

export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=${HOST_GPU_NUM} --nnodes=${HOST_NUM} \
            --node_rank=${INDEX} --master_addr=${CHIEF_IP} --master_port=${port} \
            sr_val_ddim_moe.py --config $CONFIG_PATH --ckpt $CKPT_PATH --outdir $OUTPUT_PATH --ddim_steps $sample_steps --init-img $INPUT_PATH --ddim_eta 1.0 --color_fix --n_samples 1 --data_idx `expr $num_node \* 8` --num_length ${num_length} --factor 8 --decoder_config ${decoder_config} --decoder_ckpt ${decoder_ckpt} --device 0 &

# for i in {0..0}
# do
# this_port=$(($port + ${i}))

# torchrun --nproc_per_node=${HOST_GPU_NUM} --nnodes=${HOST_NUM} \
#             --node_rank=${INDEX} --master_addr=${CHIEF_IP} --master_port=${this_port} \
#             sr_val_ddim_moe.py --config $CONFIG_PATH --ckpt $CKPT_PATH --outdir $OUTPUT_PATH --ddim_steps $sample_steps --init-img $INPUT_PATH --ddim_eta 1.0 --color_fix --n_samples 1 --data_idx `expr $num_node \* 8 + ${i}` --num_length ${num_length} --factor 8 --decoder_config ${decoder_config} --decoder_ckpt ${decoder_ckpt} --device ${i} &
# done
