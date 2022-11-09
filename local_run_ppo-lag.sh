task_name="train-ppo-lag"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
source /scratch1/miniconda3/bin/activate
conda activate cr37
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-benchmark/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-benchmark/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
cd ./interface/
nohup python train_ppo.py ../config/highD_velocity_distance_constraint/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20.yaml -n 5 -l "$log_dir" > nohup.out 2>&1 &
cd ../