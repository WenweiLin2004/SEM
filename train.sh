#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

CUDA_VISIBLE_DEVICES=${gpu_id} python robo_orchard_lab/projects/sem/robotwin/train.py \
    --config "robo_orchard_lab/projects/sem/robotwin/config_sem_robotwin.py" \
    --data_path "robo_orchard_lab/projects/sem/robotwin/data/lmdb-${task_name}-${task_config}-${expert_data_num}" \
    --task_name "${task_name}" \
    --workspace "workspace/${task_name}-${task_config}-${expert_data_num}-${seed}" 

