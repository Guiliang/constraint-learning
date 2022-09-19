task_name="train-vicrl_1"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-local-${task_name}-${launch_time}.out"
source /scratch1/miniconda3/bin/activate
conda activate me-py37
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-benchmark/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-benchmark/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
cd ./interface/
nohup python train_icrl.py ../config/mujoco_AntWall-v0/train_VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1_VaR-1e-1.yaml -n 5 -s 123 -l "$log_dir" > nohup_1.out 2>&1 &
cd ../