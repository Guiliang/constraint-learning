#!/bin/bash
#SBATCH -N 1
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=150GB
#SBATCH --job-name=VICRL
task_name="train-HighD-VICRL"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
export PATH=/pkgs/anaconda3/bin:$PATH
source /pkgs/anaconda3/bin/activate
conda activate cn-py37
cd ./interface
python train_icrl.py ../config/highD_velocity_constraint/train_VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_acbf-5e-1_no-buffer_vm-40_VaR-9e-1.yaml -n 5 -s 321 -l "$log_dir"
echo shell finish running
