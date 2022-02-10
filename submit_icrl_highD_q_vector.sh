#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=24GB
#SBATCH --job-name=PPO
task_name="train-highD-PPO"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate cr37
cd ./interface/
python train_icrl.py ../config/train_ICRL_highD.yaml -l "$log_dir"