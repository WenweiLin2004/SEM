# 1. Install
## Git Clone
```bash
cd RoboTwin/policy
git clone {repo}
```
## Prepare Python Dependency
```bash
cd policy/SEM
pip install -r requirements.txt 
```
## Prepare pre-trained weights
```bash
cd policy/SEM
mkdir pre_ckpt

# Swin-Tiny
wget https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth -O pre_ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth
python robo_orchard_lab/projects/sem/robotwin/tools/ckpt_rename.py pre_ckpt/groundingdino_swint_ogc_mmdet-822d7e9d.pth --output ./pre_ckpt
```
Download bert config and pretrain weights from [huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main).

```text
pre_ckpt
    ├──groundingdino_swint_ogc_mmdet-822d7e9d.pth
    ├──groundingdino_swint_ogc_mmdet-822d7e9d-rename.pth  # generated after rename
    └──bert-base-uncased
        ├──config.json
        ├──tokenizer_config.json
        ├──tokenizer.json
        ├──pytorch_model.bin
        ...
```
# 2. Prepare Training Data
```bash
bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# bash process_data.sh place_empty_cup demo_randomized 10
```

# 3. Train Policy
The training configuration is set at the path `robo_orchard_lab/projects/sem/robotwin/config_sem_robotwin.py`.
The trained models will be saved in the `SEM/workspace` folder.

```bash
bash train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${gpu_id}
# bash train.sh place_empty_cup demo_randomized 10 0 0

# train with multi-gpu multi-machine
bash multi_train.sh ${task_name} ${task_config} ${expert_data_num} ${seed} ${num_processes} ${gpu_ids}
# bash multi_train.sh place_empty_cup demo_randomized 10 0 4 0,1,2,3
```
# 4. Eval Policy
Move the models from `./workspace` to `./checkpoints`,and modify the `ckpt_path` parameter in both `./eval.sh` and `deploy_policy.yml`

then
```bash
sh eval.sh
```