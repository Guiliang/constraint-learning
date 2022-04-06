#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=24GB
#SBATCH --job-name=Mojuco
task_name="train-Mojuco-ICRL"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
export PATH=/pkgs/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/pkgs/mjpro150/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/galen/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
source /pkgs/anaconda3/bin/activate
conda activate cn-py37
cd ./interface
python train_icrl.py ../config/train_ICRL_HCWithPos-v0_with-action_with-buffer.yaml -s 123 -l "$log_dir"
#python train_icrl.py ../config/train_ICRL_HCWithPos-v0_with-action_with-buffer.yaml -s 321 -l "$log_dir"
#python train_icrl.py ../config/train_ICRL_HCWithPos-v0_with-action_with-buffer.yaml -s 666 -l "$log_dir"

