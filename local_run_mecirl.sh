task_name="train-mujoco-meicrl"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
export MUJOCO_PY_MUJOCO_PATH=/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch1/PycharmProjects/constraint-learning-mixture-experts/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
source /scratch1/miniconda3/bin/activate
conda activate cn-py37
pip install -e ./mujuco_environment/
cd ./interface/
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-0_robust-3e-1_advloss.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-1e1_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-3e0_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_weight-5e0_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-5e1_plr-1e-2_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-5e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-1e6_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_bi-5e2_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujuco_mixture_HCWithPos-v0/train_MEICRL_HCWithPos-v0_cbs-64_lr-5e-5_ft-2e5_exp-neg-coef-5e-1_piv-1e1_plr-1e-2_dlr-0_noisy.yaml -s 123 -n 5 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujoco_mixture_WGW-v0/train_MEICRL_WGW-v0.yaml -s 123 -n 1 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujoco_mixture_WGW-v0/train_MEICRL_WGW-v0_with_buffer.yaml -s 123 -n 1 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujoco_mixture_WGW-v0/train_MEICRL_WGW-v0_max_nu-1e0.yaml -s 123 -n 1 -l "$log_dir" > nohup.out 2>&1 &
#nohup python train_meicrl.py ../config/mujoco_mixture_WGW-v0/train_MEICRL_WGW-v0_aclr-1e0.yaml -s 123 -n 1 -l "$log_dir" > nohup.out 2>&1 &
nohup python train_meicrl.py ../config/mujoco_mixture_WGW-v0/train_MEICRL_WGW-v0_aclr-1e0_clr-1e-2.yaml -s 123 -n 1 -l "$log_dir" > nohup.out 2>&1 &
#process_id=$!
#wait $process_id
#echo shell finish running round $process_id
cd ../