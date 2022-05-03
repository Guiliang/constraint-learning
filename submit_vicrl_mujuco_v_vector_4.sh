#!/bin/bash
#SBATCH -N 1
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --partition=t4v1,t4v2,p100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem=24GB
#SBATCH --job-name=VICRL
task_name="train-Mojuco-VICRL_2"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-${task_name}-${launch_time}.out"
export PATH=/pkgs/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/pkgs/mjpro150/bin:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/h/galen/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
source /pkgs/anaconda3/bin/activate
conda activate cn-py37
pip install -e ./mujuco_environment
cd ./interface
python train_icrl.py ../config/mujoco_InvertedPendulumWall-v0/train_VICRL_InvertedPendulumWall-v0_prl-1e-2_p-9e-3-1e-3_mode-mean.yaml -n 5 -s 123 -l "$log_dir"
process_id=$!
wait $process_id
echo shell finish running
