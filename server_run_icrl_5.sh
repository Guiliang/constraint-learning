task_name="train-icrl_5"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-server-${task_name}-${launch_time}.out"
source /data/Galen/miniconda3-4.12.0/bin/activate
conda activate cn-py37
export MUJOCO_PY_MUJOCO_PATH=/data/Galen/project-constraint-learning-benchmark/constraint-learning/.mujoco/mujoco210
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/Galen/project-constraint-learning-benchmark/constraint-learning/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
pip install -e ./mujuco_environment/
cd ./interface/
export CUDA_VISIBLE_DEVICES=1
python train_icrl.py ../config/mujuco_HCWithPos-v0/train_ICRL_HCWithPos-v0_with-action_noise-5e-3.yaml -n 5 -s 123 -l "$log_dir"
python train_icrl.py ../config/mujuco_HCWithPos-v0/train_ICRL_HCWithPos-v0_with-action_noise-5e-3.yaml -n 5 -s 321 -l "$log_dir"
python train_icrl.py ../config/mujuco_HCWithPos-v0/train_ICRL_HCWithPos-v0_with-action_noise-5e-3.yaml -n 5 -s 666 -l "$log_dir"
cd ../