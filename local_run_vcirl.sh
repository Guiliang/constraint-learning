task_name="train-vicrl_1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-local-${task_name}-${launch_time}.out"
source /scratch1/miniconda3/bin/activate
conda activate cn-py37
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-benchmark/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-benchmark/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
cd ./interface/
nohup python train_icrl.py ../config/mujoco_SwimmerWithPos-v0/train_VICRL_SwmWithPos-v0_update_b-5e-1_piv-5.yaml -n 5 -s 456 -l "$log_dir" > nohup_1.out 2>&1 &
cd ../