#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=GAIL
task_name="train-highD-GAIL_1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate galen-cr37
cd ./interface/
python train_gail.py ../config/highD_distance_constraint/train_GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20.yaml -s 123 -n 5 -l "$log_dir"
