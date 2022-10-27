task_name="train-mujoco-ppo"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
source /scratch1/miniconda3/bin/activate
conda activate cn-py37
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
cd ./interface/
nohup python train_ppo.py ../config/mujuco_mixture_AntWall-v0/train_me_c-0_ppo_lag_AntWall-v0.yaml -n 5 -s 123 -l "$log_dir" > nohup-1.out 2>&1 &
#nohup python train_ppo.py ../config/mujuco_mixture_AntWall-v0/train_me_c-1_ppo_lag_AntWall-v0.yaml -n 5 -s 123 -l "$log_dir" > nohup-2.out 2>&1 &
cd ../