#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=160GB
#SBATCH --job-name=ICRL
task_name="train-highD-ICRL"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate galen-cr37
cd ./interface/
#python train_commonroad_icrl.py ../config/train_ICRL_highD_collision_constraint.yaml -s 123  -p 1 -l "$log_dir"
#python train_commonroad_icrl.py ../config/train_ICRL_highD_collision_constraint.yaml -s 123   -n 5 -l "$log_dir"
python train_commonroad_icrl.py ../config/train_ICRL_highD_velocity_constraint_no_is.yaml -s 123 -n 5 -l "$log_dir"
#python train_commonroad_icrl.py ../config/train_ICRL_highD_velocity_constraint_no_is_dim-2.yaml -s 123  -n 5 -l "$log_dir"
#python train_commonroad_icrl.py ../config/train_ICRL_highD_velocity_constraint_no_is_dim-3.yaml -s 321 -n 5 -l "$log_dir"
