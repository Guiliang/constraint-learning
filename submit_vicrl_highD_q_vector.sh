#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=160GB
#SBATCH --job-name=VICRL
task_name="train-highD-VICRL"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate galen-cr37
cd ./interface/
python train_commonroad_icrl.py ../config/train_VICRL_highD_velocity_constraint_no_is_p-1-1_dim-2.yaml -s 123 -n 5 -l "$log_dir"
