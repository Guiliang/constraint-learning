task_name="train-circle"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-local-${task_name}-${launch_time}.out"
source /scratch1/miniconda3/bin/activate
conda activate me-py37
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
cd ./interface/
nohup python train_meicrl.py ../config/others/train_MEICRL_Circle.yaml -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/others/train_MEICRL_Circle_dim2.yaml -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/others/train_MEICRL_Circle_dim3.yaml -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/others/train_MEICRL_Circle_clr-0.001.yaml -n 5 -l "$log_dir" > nohup.out 2>&1 &
cd ../