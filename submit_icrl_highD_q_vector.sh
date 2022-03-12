#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=ICRL
task_name="train-highD-ICRL"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate galen-cr37
cd ./interface/
#python train_commonroad_icrl.py ../config/train_ICRL_highD_collision_constraint.yaml -p 1 -l "$log_dir"
python train_commonroad_ppo.py ../config/train_ICRL_highD_collision_constraint.yaml  -n 5 -l "$log_dir"