#!/bin/bash
#SBATCH -N 1
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=Binary
task_name="train-Mojuco-Binary_1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
export PATH=/pkgs/anaconda3/bin:$PATH
source /pkgs/anaconda3/bin/activate
conda activate cn-py37
cd ./interface
python train_icrl.py ../config/highD_velocity_constraint/train_Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40.yaml -n 5 -s 123 -l "$log_dir"
echo shell finish running
