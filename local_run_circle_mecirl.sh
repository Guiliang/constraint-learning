task_name="train-mujoco-meicrl"
launch_time=$(date +"%H:%M-%m-%d-%y")
log_dir="log-local-${task_name}-${launch_time}.out"
source /scratch1/miniconda3/bin/activate
conda activate me-py37
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
cd ./interface/
nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64.yaml -s 321 -n 5 -l "$log_dir" > nohup.out 2>&1 &
nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64.yaml -s 666 -n 5 -l "$log_dir" > nohup.out 2>&1 &
cd ../