task_name="train-vicrl_1"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
source /scratch1/miniconda3/bin/activate
conda activate cr37
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-benchmark/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-benchmark/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install -e ./mujuco_environment/
cd ./interface/
nohup python train_icrl.py ../config/mujoco_WGW-v0/train_VICRL_WGW-v0_without-action_test-v1.yaml -n 1 -s 123 -l "$log_dir" > nohup_vicrl_1.out 2>&1 &
nohup python train_icrl.py ../config/mujoco_WGW-v0/train_VICRL_WGW-v0_without-action_test-v2.yaml -n 1 -s 123 -l "$log_dir" > nohup_vicrl_1.out 2>&1 &
nohup python train_icrl.py ../config/mujoco_WGW-v0/train_VICRL_WGW-v0_without-action_test-v3.yaml -n 1 -s 123 -l "$log_dir" > nohup_vicrl_1.out 2>&1 &
cd ../