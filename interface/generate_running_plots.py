import os

from utils.data_utils import read_running_logs, compute_moving_average, mean_std_plot_results
from utils.plot_utils import plot_curve, plot_shadow_curve


def plot_results(mean_results_moving_average, std_results_moving_average, ylim, label, method_name, save_label):
    plot_x_dict = {method_name: [i for i in range(len(mean_results_moving_average))]}
    plot_mean_y_dict = {method_name: mean_results_moving_average}
    plot_std_y_dict = {method_name: std_results_moving_average}
    # plot_curve(draw_keys=[method_name],
    #            x_dict=plot_x_dict,
    #            y_dict=plot_y_dict,
    #            ylim=ylim,
    #            xlabel='Episode',
    #            ylabel=label,
    #            title='{0}'.format(save_label),
    #            plot_name='./plot_results/{0}'.format(save_label))
    plot_shadow_curve(draw_keys=[method_name],
                      x_dict_mean=plot_x_dict,
                      y_dict_mean=plot_mean_y_dict,
                      x_dict_std=plot_x_dict,
                      y_dict_std=plot_std_y_dict,
                      ylim=ylim,
                      title='{0}'.format(save_label),
                      xlabel='Episode',
                      ylabel=label,
                      legend_size=None,
                      plot_name='./plot_results/{0}'.format(save_label), )


def generate_plots():
    file_type = "ICRL_Pos_with-buffer_with-action_crl-5e-3"
    env_id = 'HCWithPos-v0'  # 'commonroad-v1', 'HCWithPos-v0'
    modes = ['train', 'test']
    max_episodes = 10000
    for mode in modes:
        # plot_key = ['reward', 'is_collision', 'is_off_road', 'is_goal_reached', 'is_time_out']
        if env_id == 'commonroad-v1':
            max_reward = 50
            min_reward = -50
            plot_key = ['reward', 'is_collision', 'is_off_road',
                        'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed']
            plot_y_lim_dict = {'reward': (-50, 50),
                               'is_collision': (0, 1),
                               'is_off_road': (0, 1),
                               'is_goal_reached': (0, 1),
                               'is_time_out': (0, 1),
                               'avg_velocity': (20, 50),
                               'is_over_speed': (0, 1)}
            log_path_dict = {
                "PPO_highD_velocity": [
                    '../save_model/PPO-highD/train_ppo_highD_velocity_penalty-multi_env-Mar-20-2022-10:21-seed_123/',
                    '../save_model/PPO-highD/train_ppo_highD_velocity_penalty-multi_env-Mar-21-2022-05:29-seed_123/',
                ],
                "PPO_highD_no-velocity": [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty-multi_env-Mar-20-2022-10:18-seed_123/',
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty-multi_env-Mar-21-2022-05:30-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10": [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10-multi_env-Apr-02-2022-01:16-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-06-2022-11:28-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-45": [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-45-multi_env-Apr-05-2022-09:47-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-50": [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50-multi_env-Apr-05-2022-09:45-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm--45": [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm--45-multi_env-Apr-03-2022-04:07-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm--50": [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm--50-multi_env-Apr-03-2022-04:02-seed_123/'
                ],
                "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_vm-45": [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_vm-45-multi_env-Apr-03-2022-04:10-seed_123/'
                ],
                'PPO_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40': [
                    '../save_model/PPO-highD/train_ppo_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-06-2022-11:29-seed_123/'
                ],
                'PPO_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-45': [
                    '../save_model/PPO-highD/train_ppo_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-45-multi_env-Apr-04-2022-01:46-seed_123/'
                ],
                'PPO_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50': [
                    '../save_model/PPO-highD/train_ppo_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50-multi_env-Apr-04-2022-01:47-seed_123/'
                ],
                'PPO_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-5e-1': [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-5e-1-multi_env-Apr-06-2022-06:10-seed_123/'
                ],
                'PPO_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-7e-1': [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-7e-1-multi_env-Apr-06-2022-06:07-seed_123/'
                ],
                'PPO_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-9e-1': [
                    '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-50_gamma-9e-1-multi_env-Apr-06-2022-06:11-seed_123/'
                ],
                'PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40': [
                    '../save_model/PPO-Lag-highD/train_ppo_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40-multi_env-Apr-10-2022-12:45-seed_123/'
                ],
                "ICRL_highD_velocity-dim2": [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_dim-2-multi_env-Mar-26-2022-00:37/',
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_dim-2-multi_env-Mar-26-2022-08:02/'
                ],
                "ICRL_highD_velocity-dim2-buff": [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_dim-2-multi_env-Mar-28-2022-06:56/',
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_dim-2-multi_env-Mar-28-2022-09:44/'
                ],
                "ICRL_highD_velocity-dim3-buff": [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_dim-3-multi_env-Mar-30-2022-06:05/',
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_dim-3-multi_env-Mar-31-2022-00:47-seed_321//'
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dim-2': [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dim-2-multi_env-Apr-07-2022-23:34-seed_123/'
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_bs--1_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-08-2022-01:12-seed_123/'
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-08-2022-01:12-seed_123/',
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-11-2022-00:31-seed_123/'
                ],
                'ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-11-2022-00:30-seed_123/',
                ],
                'ICRL_highD_velocity_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-11-2022-11:53-seed_123/',
                ],
                'ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2': [
                    '../save_model/ICRL-highD/train_ICRL_highD_velocity_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2-multi_env-Apr-11-2022-04:12-seed_123/',
                ],
                "VICRL_highD_velocity-dim2-buff": [
                    '../save_model/VICRL-highD/train_VICRL_highD_velocity_constraint_no_is_p-1-1_dim-2-multi_env-Mar-31-2022-06:36-seed_123/'
                ],
            }
        elif env_id == 'HCWithPos-v0':
            max_reward = 10000
            min_reward = -10000
            plot_key = ['reward', 'constraint']
            plot_y_lim_dict = {'reward': (0, 6000),
                               'constraint': (0, 1)}
            log_path_dict = {
                "PPO_Pos": [
                    '../save_model/PPO-HC/train_ppo_HCWithPos-v0-multi_env-Apr-06-2022-05:18-seed_123/',
                    '../save_model/PPO-HC/train_ppo_HCWithPos-v0-multi_env-Apr-07-2022-10:23-seed_321/',
                    '../save_model/PPO-HC/train_ppo_HCWithPos-v0-multi_env-Apr-07-2022-05:13-seed_666/'
                ],
                "PPO_lag_Pos": [
                    '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-11-2022-07:16-seed_123/',
                    '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-11-2022-07:18-seed_321/',
                    '../save_model/PPO-Lag-HC/train_ppo_lag_HCWithPos-v0-multi_env-Apr-11-2022-07:18-seed_666/'
                ],
                "ICRL_Pos": [
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-06-2022-10:56-seed_123/',
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-06-2022-10:58-seed_321/',
                    # '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-06-2022-10:59-seed_666/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-07:36-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-05:43-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-07:16-seed_666/'
                ],
                "ICRL_Pos_crl-5e-3": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_crl-5e-3-multi_env-Apr-12-2022-12:46-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_crl-5e-3-multi_env-Apr-12-2022-12:49-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_crl-5e-3-multi_env-Apr-12-2022-13:01-seed_666/',
                ],
                "ICRL_Pos_with-buffer": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-12:19-seed_321/',
                ],
                "ICRL_Pos_with-buffer_with-action": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer-Apr-06-2022-10:56-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer-Apr-06-2022-10:58-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer-Apr-06-2022-10:59-seed_666/',
                ],
                "ICRL_Pos_with-buffer_with-action_crl-5e-3": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer_crl-5e-3-multi_env-Apr-12-2022-12:43-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer_crl-5e-3-multi_env-Apr-12-2022-12:49-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_with-buffer_crl-5e-3-multi_env-Apr-12-2022-13:02-seed_666/',
                ],
                "ICRL_Pos_with-buffer-100k_with-action": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-23:50-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-23:50-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0-Apr-05-2022-23:51-seed_666/',
                ],
                "ICRL_Pos_with-action": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-Apr-06-2022-10:56-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-Apr-06-2022-10:58-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action-Apr-06-2022-10:59-seed_666/',
                ],
                "ICRL_Pos_with-action_crl-5e-3": [
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-12-2022-12:43-seed_123/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-12-2022-12:49-seed_321/',
                    '../save_model/ICRL-HC/train_ICRL_HCWithPos-v0_with-action_crl-5e-3-multi_env-Apr-12-2022-13:02-seed_666/',
                ],
                "VICRL_Pos": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_p-1-1_no_is-multi_env-Apr-07-2022-10:23-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_p-1-1_no_is-multi_env-Apr-07-2022-10:25-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_p-1-1_no_is-multi_env-Apr-07-2022-10:27-seed_666/',
                ],
                "VICRL_Pos_with-action": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_p-1-1_no_is-multi_env-Apr-07-2022-10:23-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_p-1-1_no_is-multi_env-Apr-07-2022-10:26-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_p-1-1_no_is-multi_env-Apr-07-2022-10:27-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-1_no_is-multi_env-Apr-07-2022-10:23-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-1_no_is-multi_env-Apr-07-2022-10:26-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-1_no_is-multi_env-Apr-07-2022-10:27-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action_p-1-9": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-9_no_is-multi_env-Apr-08-2022-01:00-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-9_no_is-multi_env-Apr-08-2022-01:01-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1-9_no_is-multi_env-Apr-08-2022-01:06-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action_p-1e-1-9e-1": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1e-1-9e-1_no_is-multi_env-Apr-08-2022-01:00-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1e-1-9e-1_no_is-multi_env-Apr-08-2022-01:01-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-1e-1-9e-1_no_is-multi_env-Apr-08-2022-01:06-seed_666/',
                ],
                "VICRL_Pos_with-buffer_with-action_p-9-1": [
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-08-2022-04:32-seed_123/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-08-2022-04:32-seed_321/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-08-2022-04:33-seed_666/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-07:23-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-07:22-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9-1_no_is-multi_env-Apr-11-2022-06:54-seed_666/'
                ],
                "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1": [
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-08-2022-04:32-seed_123/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-08-2022-04:33-seed_321/',
                    # '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-08-2022-04:33-seed_666/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-11-2022-06:56-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-11-2022-07:00-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_no_is-multi_env-Apr-11-2022-07:01-seed_666/'
                ],
                "VICRL_Pos_with-buffer_with-action_p-9e-2-1e-2": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_no_is-multi_env-Apr-11-2022-11:17-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_no_is-multi_env-Apr-11-2022-11:18-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-2-1e-2_no_is-multi_env-Apr-11-2022-11:21-seed_666/'
                ],
                "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-11-2022-11:22-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-11-2022-11:18-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is-multi_env-Apr-11-2022-11:21-seed_666/'
                ],
                "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3_bs-64-1e3": [
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_bs-64-1e3_no_is-multi_env-Apr-11-2022-11:17-seed_123/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_bs-64-1e3_no_is-multi_env-Apr-11-2022-11:18-seed_321/',
                    '../save_model/VICRL-HC/train_VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_bs-64-1e3_no_is-multi_env-Apr-11-2022-11:21-seed_666/'
                ],
            }
        else:
            raise ValueError("Unknown env id {0}".format(env_id))
        all_results = []
        for log_path in log_path_dict[file_type]:
            monitor_path_all = []
            if mode == 'train':
                run_files = os.listdir(log_path)
                for file in run_files:
                    if 'monitor' in file:
                        monitor_path_all.append(log_path + file)
            else:
                monitor_path_all.append(log_path + 'test/test.monitor.csv')

            # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
            results = read_running_logs(monitor_path_all=monitor_path_all, read_keys=plot_key,
                                        max_reward=max_reward, min_reward=min_reward)
            all_results.append(results)

        mean_results, std_results = mean_std_plot_results(all_results)

        # if not os.path.exists('./plot_results/' + log_path.split('/')[3]):
        #     os.mkdir('./plot_results/' + log_path.split('/')[3])
        if not os.path.exists('./plot_results/' + file_type):
            os.mkdir('./plot_results/' + file_type)

        for idx in range(len(plot_key)):
            mean_results_moving_average = compute_moving_average(result_all=mean_results[plot_key[idx]],
                                                                 average_num=100)
            std_results_moving_average = compute_moving_average(result_all=std_results[plot_key[idx]],
                                                                average_num=100)
            if max_episodes:
                mean_results_moving_average = mean_results_moving_average[:max_episodes]
                std_results_moving_average = std_results_moving_average[:max_episodes]
            plot_results(mean_results_moving_average,
                         std_results_moving_average,
                         label=plot_key[idx],
                         method_name=file_type,
                         ylim=plot_y_lim_dict[plot_key[idx]],
                         save_label=file_type + '/' + plot_key[idx] + '_' + mode)


if __name__ == "__main__":
    generate_plots()
