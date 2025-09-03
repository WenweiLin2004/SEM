#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}

python robo_orchard_lab/robo_orchard_lab/dataset/robotwin/robotwin_packer.py \
    --input_path "../../data/" \
    --output_path "robo_orchard_lab/projects/sem/robotwin/data/lmdb-${task_name}-${task_config}-${expert_data_num}" \
    --task_name ${task_name} \
    --config_name ${task_config} \
    --max_episodes ${expert_data_num}
