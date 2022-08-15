#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=VICRL
task_name="train-highD-VICRL_1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate cn-py37
cd ./interface/
python train_icrl.py ../config/highD_velocity_constraint/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_data-1e-1.yaml -s 321 -n 5 -l "$log_dir"