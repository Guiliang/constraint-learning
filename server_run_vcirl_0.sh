#!/bin/bash
task_name="train-vicrl_0"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-server-${task_name}-${launch_time}.out"
source /data/Galen/miniconda3-4.12.0/bin/activate
source activate cn-py37
export MUJOCO_PY_MUJOCO_PATH=/data/Galen/project-constraint-learning-benchmark/constraint-learning/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/Galen/project-constraint-learning-benchmark/constraint-learning/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install -e ./mujuco_environment/
cd ./interface/
export CUDA_VISIBLE_DEVICES=0
#python train_icrl.py ../config/mujoco_Circle-v0/train_VICRL_Circle_dim2.yaml -n 5 -s 123 -l "$log_dir"
#python train_icrl.py ../config/mujoco_Circle-v0/train_VICRL_Circle_dim2_acbf-8e-1.yaml -n 5 -s 123 -l "$log_dir"
python train_icrl.py ../config/mujoco_Circle-v0/train_VICRL_Circle_dim2_acbf-85e-2.yaml -n 5 -s 123 -l "$log_dir"
python train_icrl.py ../config/mujoco_Circle-v0/train_VICRL_Circle_dim2_piv-1e1.yaml -n 5 -s 123 -l "$log_dir"
python train_icrl.py ../config/mujoco_Circle-v0/train_VICRL_Circle_dim2_piv-2e1.yaml -n 5 -s 123 -l "$log_dir"
cd ../