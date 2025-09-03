#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
num_processes=${5}
gpu_ids=${6}

accelerate launch \
    --multi-gpu \
    --num_processes ${num_processes} \
    --gpu_ids ${gpu_ids} \
    robo_orchard_lab/projects/sem/robotwin/train.py \
    --config "robo_orchard_lab/projects/sem/robotwin/config_sem_robotwin.py" \
    --data_path "robo_orchard_lab/projects/sem/robotwin/data/lmdb-${task_name}-${task_config}-${expert_data_num}" \
    --task_name "${task_name}" \
    --workspace "workspace/${task_name}-${task_config}-${expert_data_num}-${seed}" 