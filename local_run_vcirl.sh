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
#nohup python train_icrl.py ../config/mujoco_WGW-v0/train_VICRL_WGW-v0_without-action_by_games_max-nu-1_recon_obs_with-buffer-setting1.yaml -n 1 -s 123 -l "$log_dir" > nohup_vicrl.out 2>&1 &
nohup python train_icrl.py ../config/mujuco_HCWithPos-v0/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is.yaml -n 5 -s 123 -l "$log_dir" > nohup_vicrl.out 2>&1 &
nohup python train_icrl.py ../config/mujuco_HCWithPos-v0/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is.yaml -n 5 -s 123 -l "$log_dir" > nohup_vicrl.out 2>&1 &
cd ../