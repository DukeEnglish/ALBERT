#!/usr/bin/env bash
#########################################################################
# File Name: issue.sh
# Author: Junyi Li
# Personal page: dukeenglish.github.io
# Created Time: 21:00:20 2020-01-11
########################################################################
set -ex

OUTPUT_DIR="CLUE_csl"
# export CUDA_VISIBLE_DEVICES="0"
ALBERT_ROOT=~/project/ALBERT
MODEL=/home/ljy/NLP_MODEL/albert_base
# To start from a custom pretrained checkpoint, set ALBERT_HUB_MODULE_HANDLE
# below to an empty string and set INIT_CHECKPOINT to your checkpoint path.
ALBERT_HUB_MODULE_HANDLE="" # "https://tfhub.dev/google/albert_base/1"
INIT_CHECKPOINT="${MODEL}/model.ckpt-best"

# INIT_CHECKPOINT="${ALBERT_ROOT}/albert_base//model.ckpt-best.data-00000-of-00001"



function run_task() {
  COMMON_ARGS="--output_dir="${OUTPUT_DIR}/$1" --data_dir="/home/ljy/CLUE/baselines/models_pytorch/classifier_pytorch/CLUEdatasets/$1/" --vocab_file="${MODEL}/vocab_chinese.txt"  --do_lower_case --max_seq_length=128 --optimizer=adamw --task_name=$1 --warmup_step=$2 --learning_rate=$3 --train_step=$4 --save_checkpoints_steps=$5 --train_batch_size=$6"
  python3 -m run_classifier \
      ${COMMON_ARGS} \
      --do_train \
      --nodo_eval \
      --nodo_predict \
      --albert_config_file="${MODEL}/albert_config.json" \
      --init_checkpoint="${INIT_CHECKPOINT}"
      # --albert_config_file="${ALBERT_ROOT}/albert_base/albert_config.json" \
      # # --albert_hub_module_handle="${ALBERT_HUB_MODULE_HANDLE}" \
      # --init_checkpoint="${INIT_CHECKPOINT}"
   python3 -m run_classifier \
       ${COMMON_ARGS} \
       --nodo_train \
       --do_eval \
       --albert_config_file="${MODEL}/albert_config.json" \
       --do_predict
  #     # --albert_config_file="${ALBERT_ROOT}/albert_base/albert_config.json" \

  #   #   --albert_hub_module_handle="${ALBERT_HUB_MODULE_HANDLE}" \
}


# --task_name=$1 --warmup_step=$2 --learning_rate=$3 
# --train_step=$4 --save_checkpoints_steps=$5 --train_batch_size=$6"
# run_task csl 200 3e-5 800 100 32
# run_task wsc 200 3e-5 1500 100 16

run_task tnews 100 3e-5 800 100 128
run_task xnli 100 3e-5 800 100 16
run_task afqmc 100 3e-5 800 100 16
run_task cmnli 100 3e-5 800 100 128
run_task iflytek 100 3e-5 800 100 128



