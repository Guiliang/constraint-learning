#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --mem=120GB
#SBATCH --job-name=collect
task_name="collect_data-highD"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
source /h/galen/miniconda3/bin/activate
conda activate galen-cr37
cd ./interface/
python generate_data_for_icrl_commonroad_evaluation.py -n 5