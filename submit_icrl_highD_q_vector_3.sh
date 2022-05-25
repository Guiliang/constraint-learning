#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=ICRL
task_name="train-highD-ICRL_3"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate galen-cr37
cd ./interface/
python train_icrl.py ../config/highD_distance_constraint/train_ICRL_highD_slo_distance_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20.yaml -s 123 -n 5 -l "$log_dir"
