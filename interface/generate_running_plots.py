import copy
import os

import numpy as np

from interface.plot_results.plot_results_dirs import get_plot_results_dir
from utils.data_utils import read_running_logs, compute_moving_average, mean_std_plot_results, \
    mean_std_plot_valid_rewards, mean_std_test_results
from utils.plot_utils import plot_curve, plot_shadow_curve


def plot_results(mean_results_moving_avg_dict,
                 std_results_moving_avg_dict,
                 episode_plots,
                 ylim, label, method_names,
                 save_label,
                 legend_dict=None,
                 linestyle_dict=None,
                 legend_size=20,
                 axis_size=None,
                 img_size=None,
                 title=None):
    plot_mean_y_dict = {}
    plot_std_y_dict = {}
    plot_x_dict = {}
    for method_name in method_names:
        # plot_x_dict.update({method_name: [i for i in range(len(mean_results_moving_avg_dict[method_name]))]})
        plot_x_dict.update({method_name: episode_plots[method_name]})
        plot_mean_y_dict.update({method_name: mean_results_moving_avg_dict[method_name]})
        plot_std_y_dict.update({method_name: std_results_moving_avg_dict[method_name]})
    if save_label is not None:
        plot_name = './plot_results/{0}'.format(save_label)
    else:
        plot_name = None
    plot_shadow_curve(draw_keys=method_names,
                      x_dict_mean=plot_x_dict,
                      y_dict_mean=plot_mean_y_dict,
                      x_dict_std=plot_x_dict,
                      y_dict_std=plot_std_y_dict,
                      img_size=img_size if img_size is not None else (6, 5.8),
                      ylim=ylim,
                      title=title,
                      xlabel='Episode',
                      ylabel=label,
                      legend_dict=legend_dict,
                      legend_size=legend_size,
                      linestyle_dict=linestyle_dict,
                      axis_size=axis_size if axis_size is not None else 18,
                      title_size=20,
                      plot_name=plot_name, )


def generate_plots():
    axis_size = None
    save = True

    # env_id = 'HCWithPos-v0'
    # max_episodes = 6000
    # average_num = 100
    # max_reward = 10000
    # min_reward = 0
    # plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    # label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
    # label_key = [None, None, None, None]
    # img_size = None
    # save = True
    # title = 'Blocked Half-Cheetah'
    # constraint_keys = ['constraint']
    # plot_y_lim_dict = {'reward': (0, 7000),
    #                    'reward_nc': (0, 5000),
    #                    'constraint': (0, 1.1),
    #                    'reward_valid': (0, 5000),
    #                    }
    # method_names_labels_dict = {
    #     "GAIL_HCWithPos-v0_with-action": 'GACL',  # 'GAIL',
    #     "Binary_HCWithPos-v0_with-action": 'BC2L',  # 'Binary',
    #     "ICRL_Pos_with-action": 'MECL',  # 'ICRL',
    #     "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3": "VCIRL-SR",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-1e-1": "VCIRL2",
    #     "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-5e-1": "VCIRL-VaR",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_VAR-9e-1": "VCIRL4",
    #     # "PPO_Pos": 'PPO',
    #     # "PPO_lag_Pos": 'PPO_lag',
    # }
    # # ================= rebuttal ====================
    # max_episodes = 5000
    # img_size = None
    # save = False
    # title = 'Dataset with Noise'
    # method_names_labels_dict = {
    #     # "VICRL_HCWithPos-v0_with_action_p-9e-1-1e-1_no_is_reset-setting1": "VCIRL1",
    #     # "VICRL_HCWithPos-v0_with_action_p-9e-1-1e-1_no_is_reset-setting2": "VCIRL2",
    #     # "VICRL_HCWithPos-v0_with_action_p-9e-1-1e-1_no_is_reset-setting3": "VCIRL3",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1_no_is": "VCIRL1",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1_no_is": "VCIRL-0.3",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1_no_is": "VCIRL3",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-1e-1-b_no_is": "VCIRL1",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-3e-1-b_no_is": "VCIRL2",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_data-5e-1-b_no_is": "VCIRL-0.5",
    #     "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-1": "Ram-1",
    #     "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-8e-1": "Ram-0.8",
    #     "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-5e-1": "Ram-0.5",
    #     "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random-2e-1": "Ram-0.2",
    #     "VICRL_Pos_with-buffer_with-action_p-9e-1-1e-1_clr-5e-3": "Ram-0",
    #     # "VICRL_HCWithPos-v0_with_action_p-1-1_no_is_hard": "VCIRL1",
    #     # "VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_hard": "VCIRL2",
    #     # 'VICRL_HCWithPos-v0_with_action_with_buffer_p-9e-1-1e-1_clr-5e-3_no_is_random': 'VCIRL_Random',
    #     # "PPO_Pos": 'PPO',
    #     "PPO_lag_Pos": 'PPO_lag',
    # }

    # env_id = 'AntWall-V0'
    # max_episodes = 15000
    # average_num = 300
    # title = 'Blocked Ant'
    # max_reward = float('inf')
    # min_reward = 0
    # # plot_key = ['reward', 'constraint', 'reward_valid', 'reward_nc']
    # plot_key = ['reward', 'constraint', 'reward_valid']
    # # label_key = ['reward', 'Constraint Violation Rate', 'reward_valid', 'reward_nc']
    # label_key = [None, None, None, None]
    # img_size = (7, 5.9)
    # save = True
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None}
    # constraint_keys = ['constraint']
    # method_names_labels_dict = {
    #     "GAIL_AntWall-v0_with-action": 'GACL',  # 'GAIL',
    #     "Binary_AntWall-v0_with-action_nit-50": 'BC2L',  # 'Binary',
    #     "ICRL_AntWall_with-action_nit-50": 'MECL',  # 'ICRL',
    #     "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1": "VCIRL-SR",
    #     # "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1_VaR-1e-1": "VCIRL2",
    #     # "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1_VaR-5e-1": "VCIRL3",
    #     "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1_VaR-7e-1": "VCIRL-VaR",
    #     # "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1_VaR-9e-1": "VCIRL5",
    #     # VICRL_AntWall-v0_with-action_no_is_nit-50_p-9-1, VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1, VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-2-1e-2
    #     # "PPO-AntWall": 'PPO',
    #     # "PPO-Lag-AntWall": 'PPO_lag',
    # }
    # ================= rebuttal ====================
    # max_episodes = 20000
    # img_size = (6.7, 5.6)
    # save = False
    # plot_key = ['reward', 'constraint', 'reward_valid']
    # method_names_labels_dict = {
    #     "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1": "VCIRL",
    #     "VICRL_AntWall-v0_with-action_no_is_nit-50_p-9e-1-1e-1_hard": "VCIRL_Hard",
    #     "PPO-AntWall": 'PPO',
    #     "PPO-Lag-AntWall_piv-5e-3": 'PPO_lag1',
    #     "PPO-Lag-AntWall_plr-1e-3": 'PPO_lag2',
    #     "PPO-Lag-AntWall": 'PPO_lag',
    # }

    # env_id = 'InvertedPendulumWall-v0'
    # max_episodes = 80000
    # average_num = 2000
    # title = 'Biased Pendulumn'
    # max_reward = float('inf')
    # min_reward = 0
    # plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    # label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
    # label_key = [None, None, None, None]
    # img_size = None
    # save = False
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None}
    # constraint_keys = ['constraint']
    # method_names_labels_dict = {
    #     # "GAIL_PendulumWall": 'GACL',  # 'GAIL',
    #     # "Binary_PendulumWall": 'BC2L',  # 'Binary',
    #     # "ICRL_Pendulum": 'MECL',  # 'ICRL',
    #     "VICRL_PendulumWall": 'VCIRL-SR',
    #     'VICRL_PendulumWall_VaR-1e-1': 'VCIRL2',
    #     'VICRL_PendulumWall_VaR-5e-1': 'VCIRL3',
    #     'VICRL_PendulumWall_VaR-7e-1': 'VCIRL4',
    #     'VICRL_PendulumWall_VaR-9e-1': 'VCIRL5',
    #     # "PPO_Pendulum": 'PPO',
    #     # "PPO_lag_Pendulum": 'PPO_lag',
    # }

    # env_id = 'WalkerWithPos-v0'
    # max_episodes = 40000
    # average_num = 2000
    # title = 'Blocked Walker'
    # max_reward = float('inf')
    # min_reward = 0
    # plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    # label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
    # label_key = [None, None, None, None]
    # # plot_y_lim_dict = {'reward': (0, 700),
    # #                    'reward_nc': (0, 700),
    # #                    'constraint': (0, 1)}
    # img_size = None
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None}
    # constraint_keys = ['constraint']
    # method_names_labels_dict = {
    #     "GAIL_Walker": 'GACL',  # 'GACL'
    #     "Binary_Walker": 'BC2L',  # 'Binary
    #     "ICRL_Walker": 'MECL',  # 'ICRL',
    #     # "VICRL_Walker-v0_p-9e-3-1e-3": 'VICRL',
    #     "VICRL_Walker-v0_p-9e-3-1e-3_cl-64-64": 'VCIRL',
    #     # "PPO_Walker": 'PPO',
    #     "PPO_lag_Walker": 'PPO_lag',
    # }

    # env_id = 'SwimmerWithPos-v0'
    # max_episodes = 10000
    # average_num = 200
    # title = 'Blocked Swimmer'
    # max_reward = float('inf')
    # min_reward = 0
    # plot_key = ['reward', 'reward_nc', 'constraint', 'reward_valid']
    # label_key = ['reward', 'reward_nc', 'Constraint Violation Rate', 'reward_valid']
    # label_key = [None, None, None, None]
    # img_size = None
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'constraint': None,
    #                    'reward_valid': None}
    # constraint_keys = ['constraint']
    # method_names_labels_dict = {
    #     "GAIL_SwmWithPos-v0": 'GACL',
    #     "Binary_SwmWithPos-v0_update_b-5e-1": 'BC2L',
    #     "ICRL_SwmWithPos-v0_update_b-5e-1": 'MECL',
    #     "VICRL_SwmWithPos-v0_update_b-5e-1_piv-5": 'VCIRL',
    #     # "ppo_SwmWithPos-v0_update_b-5e-1": 'PPO',
    #     "ppo_lag_SwmWithPos-v0_update_b-5e-1": 'PPO_lag',
    # }

    env_id = 'highD_velocity_constraint'
    max_episodes = 5000
    average_num = 200
    max_reward = 50
    min_reward = -50
    axis_size = 20
    img_size = [8.5, 6.5]
    save = False
    title = 'HighD Velocity Constraint'
    constraint_keys = ['is_over_speed']
    plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
                'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'success_rate']
    label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
                 'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Speed Constraint Violation Rate',
                 'Success Rate']
    plot_y_lim_dict = {'reward': None,
                       'reward_nc': None,
                       'reward_valid': None,
                       'is_collision': None,
                       'is_off_road': None,
                       'is_goal_reached': None,
                       'is_time_out': None,
                       'avg_velocity': None,
                       'is_over_speed': None,
                       'success_rate': None}
    bound_results = {
        'reward': 50,
        'reward_nc': 50,
        'reward_valid': 50,
        'is_collision': 0,
        'is_off_road': 0,
        'is_goal_reached': 0,
        'is_time_out': 0,
        'is_over_speed': 0,
        'success_rate': 1,
    }
    method_names_labels_dict = {
        "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": 'GACL',
        "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": 'BC2L',
        "ICRL_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": 'MECL',
        "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": "VCIRL-SR",
        "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_no-buffer_vm-40": "VCIRL2",
        "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40": "VCIRL3",
        "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4-plr-5e-2_no-buffer_vm-40": "VCIRL4",
        "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4-plr-5e-3_no-buffer_vm-40": "VCIRL5",
        # "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": 'PPO',
        "PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": 'PPO_lag',
        "Bound": 'Bound'
    }

    # env_id = 'highD_velocity_constraint_dim2'
    # max_episodes = 5000
    # average_num = 200
    # max_reward = 50
    # min_reward = -50
    # axis_size = 20
    # img_size = [8.5, 6.5]
    # title = 'Simplified HighD Velocity Constraint'
    # constraint_keys = ['is_over_speed']
    # plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
    #             'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'success_rate']
    # label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
    #              'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Speed Constraint Violation Rate',
    #              'Success Rate']
    # # plot_y_lim_dict = {'reward': (-50, 50),
    # #                    'reward_nc': (0, 50),
    # #                    'is_collision': (0, 1),
    # #                    'is_off_road': (0, 1),
    # #                    'is_goal_reached': (0, 1),
    # #                    'is_time_out': (0, 1),
    # #                    'avg_velocity': (20, 50),
    # #                    'is_over_speed': (0, 1)}
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'reward_valid': None,
    #                    'is_collision': None,
    #                    'is_off_road': None,
    #                    'is_goal_reached': None,
    #                    'is_time_out': None,
    #                    'avg_velocity': None,
    #                    'is_over_speed': None,
    #                    'success_rate': None}
    # bound_results = {
    #     'reward': 50,
    #     'reward_nc': 50,
    #     'reward_valid': 50,
    #     'is_collision': 0,
    #     'is_off_road': 0,
    #     'is_goal_reached': 0,
    #     'is_time_out': 0,
    #     'is_over_speed': 0,
    #     'success_rate': 1,
    # }
    # method_names_labels_dict = {
    #     "GAIL_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40": 'GACL',
    #     "Binary_highD_velocity_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": 'BC2L',
    #     "ICRL_highD_velocity_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": 'MECL',
    #     "VICRL_highD_velocity_constraint_p-9e-1-1e-1_no_is_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dim-2": "VCIRL",
    #     # "PPO_highD_no-velocity_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": 'PPO',
    #     "PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40": 'PPO_lag',
    #     "Bound": 'Bound'
    # }

    # env_id = 'highD_velocity_constraint_vm-30'
    # method_names_labels_dict = {
    #     "ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30": 'PPO',
    #     "PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-30": 'PPO_lag',
    # }

    # env_id = 'highD_velocity_constraint_vm-35'
    # method_names_labels_dict = {
    #     "ppo_highD_no_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35": 'PPO',
    #     "PPO_lag_highD_velocity_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-35": 'PPO_lag',
    # }

    # env_id = 'highD_distance_constraint'
    # max_episodes = 5000
    # average_num = 200
    # max_reward = 50
    # min_reward = -50
    # axis_size = 20
    # img_size = [8.5, 6.5]
    # title = 'HighD Distance Constraint'
    # constraint_keys = ['is_too_closed']
    # plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
    #             'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'avg_distance',
    #             'is_too_closed', 'success_rate']
    # label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
    #              'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate', 'Avg. Distance',
    #              'Distance Constraint Violation Rate', 'Success Rate']
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'reward_valid': None,
    #                    'is_collision': None,
    #                    'is_off_road': None,
    #                    'is_goal_reached': None,
    #                    'is_time_out': None,
    #                    'avg_velocity': None,
    #                    'avg_distance': None,
    #                    'is_over_speed': None,
    #                    'is_too_closed': None,
    #                    'success_rate': None}
    # # plot_y_lim_dict = {'reward': (-50, 50),
    # #                    'reward_nc': (0, 50),
    # #                    'is_collision': (0, 1),
    # #                    'is_off_road': (0, 1),
    # #                    'is_goal_reached': (0, 1),
    # #                    'is_time_out': (0, 1),
    # #                    'avg_velocity': (20, 50),
    # #                    'is_over_speed': (0, 1),
    # #                    'avg_distance': (50, 100),
    # #                    'is_too_closed': (0, 0.5)}
    # bound_results = {
    #     'reward': 50,
    #     'reward_nc': 50,
    #     'reward_valid': 50,
    #     'is_collision': 0,
    #     'is_off_road': 0,
    #     'is_goal_reached': 0,
    #     'is_time_out': 0,
    #     'is_too_closed': 0,
    #     'is_over_speed': 0,
    #     'success_rate': 1
    # }
    # method_names_labels_dict = {
    #     'GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20': 'GACL',
    #     'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20': 'BC2L',
    #     'ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20': 'MECL',
    #     'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20': 'VCIRL',
    #     # "ppo_highD_no_slo_distance_dm-20": 'PPO',
    #     "ppo_lag_highD_no_slo_distance_dm-20": 'PPO_lag',
    #     "Bound": 'Bound'
    # }

    # env_id = 'highD_distance_constraint_dim6'
    # max_episodes = 5000
    # average_num = 200
    # max_reward = 50
    # min_reward = -50
    # axis_size = 20
    # img_size = [8.5, 6.5]
    # title = 'Simplified HighD Distance Constraint'
    # constraint_keys = ['is_too_closed']
    # plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
    #             'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'avg_distance',
    #             'is_too_closed', 'success_rate']
    # label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
    #              'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate', 'Avg. Distance',
    #              'Distance Constraint Violation Rate', 'Success Rate']
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'reward_valid': None,
    #                    'is_collision': None,
    #                    'is_off_road': None,
    #                    'is_goal_reached': None,
    #                    'is_time_out': None,
    #                    'avg_velocity': None,
    #                    'avg_distance': None,
    #                    'is_over_speed': None,
    #                    'is_too_closed': None,
    #                    'success_rate': None}
    # # plot_y_lim_dict = {'reward': (-50, 50),
    # #                    'reward_nc': (0, 50),
    # #                    'is_collision': (0, 1),
    # #                    'is_off_road': (0, 1),
    # #                    'is_goal_reached': (0, 1),
    # #                    'is_time_out': (0, 1),
    # #                    'avg_velocity': (20, 50),
    # #                    'is_over_speed': (0, 1),
    # #                    'avg_distance': (50, 100),
    # #                    'is_too_closed': (0, 0.5)}
    # bound_results = {
    #     'reward': 50,
    #     'reward_nc': 50,
    #     'reward_valid': 50,
    #     'is_collision': 0,
    #     'is_off_road': 0,
    #     'is_goal_reached': 0,
    #     'is_time_out': 0,
    #     'is_too_closed': 0,
    #     'is_over_speed': 0,
    #     'success_rate': 1,
    # }
    # method_names_labels_dict = {
    #     "GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-20_dim-6": 'GACL',
    #     "Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": 'BC2L',
    #     "ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": 'MECL',
    #     "VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-20_dim-6": 'VCIRL',
    #     # "ppo_highD_no_slo_distance_dm-20": 'PPO',
    #     "ppo_lag_highD_no_slo_distance_dm-20": 'PPO_lag',
    #     "Bound": 'Bound'
    # }
    #
    # env_id = 'highD_velocity_distance_constraint'
    # max_episodes = 5000
    # average_num = 100
    # max_reward = 50
    # min_reward = -50
    # axis_size = 20
    # img_size = [9, 6.5]
    # save = True
    # title = 'HighD Velocity and Distance Constraint'
    # constraint_keys = ['is_too_closed', 'is_over_speed']
    # plot_key = ['reward', 'reward_nc', 'reward_valid', 'is_collision', 'is_off_road',
    #             'is_goal_reached', 'is_time_out', 'avg_velocity', 'is_over_speed', 'avg_distance',
    #             'is_too_closed', 'success_rate']
    # label_key = ['Rewards', 'Feasible Rewards', 'Feasible Rewards', 'Collision Rate', 'Off Road Rate',
    #              'Goal Reached Rate', 'Time Out Rate', 'Avg. Velocity', 'Over Speed Rate', 'Avg. Distance',
    #              'Distance Constraint Violation Rate', 'Success Rate']
    # plot_y_lim_dict = {'reward': None,
    #                    'reward_nc': None,
    #                    'reward_valid': None,
    #                    'is_collision': None,
    #                    'is_off_road': None,
    #                    'is_goal_reached': None,
    #                    'is_time_out': None,
    #                    'avg_velocity': None,
    #                    'avg_distance': None,
    #                    'is_over_speed': None,
    #                    'is_too_closed': None,
    #                    'success_rate': None}
    #
    # bound_results = {
    #     'reward': 50,
    #     'reward_nc': 50,
    #     'reward_valid': 50,
    #     'is_collision': 0,
    #     'is_off_road': 0,
    #     'is_goal_reached': 0,
    #     'is_time_out': 0,
    #     'is_too_closed': 0,
    #     'is_over_speed': 0,
    #     'success_rate': 1,
    # }
    # method_names_labels_dict = {
    #     # "ppo_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20": 'PPO',
    #     "ppo_lag_highD_velocity_distance_penalty_bs--1_fs-5k_nee-10_lr-5e-4_vm-40_dm-20": 'PPO_lag',
    #     # 'GAIL_velocity_distance_constraint_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20': 'GACL',
    #     # 'Binary_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20': 'BC2L',
    #     # 'ICRL_highD_velocity_distance_constraint_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20': 'MECL',
    #     # 'VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_no-buffer_vm-40_dm-20': 'VCIRL',
    #     # 'VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_no-buffer_vm-40_dm-20': 'VCIRL1',
    #     # 'VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-5e-4_no-buffer_vm-40_dm-20': 'VCIRL2',
    #     # 'VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_clay-64-64_no-buffer_vm-40_dm-20': 'VCIRL3',
    #     # 'VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-4_clay-64-64_no-buffer_vm-40_dm-20': 'VCIRL',
    #     # 'VICRL_highD_velocity_distance_constraint_p-9-1_no_is_bs--1-5e2_fs-5k_nee-10_lr-1e-3_clay-64-64-64_no-buffer_vm-40_dm-20': 'VCIRL5',
    #     # "Bound": 'Bound'
    # }

    # env_id = 'highD_distance_constraint_dm-40'
    # method_names_labels_dict = {
    #     "ppo_highD_no_slo_distance_dm-40": 'PPO',
    #     "ppo_lag_highD_no_slo_distance_dm-40": 'PPO_lag',
    # }
    #
    # env_id = 'highD_distance_constraint_dm-60'
    # method_names_labels_dict = {
    #     # 'GAIL_highD_slo_distance_constraint_no_is_bs--1--1_lr-5e-4_no-buffer_dm-60': 'GACL',
    #     # 'Binary_highD_slo_distance_constraint_no_is_bs--1-1e3_nee-10_lr-5e-4_no-buffer_dm-60': 'BC2L',
    #     # 'ICRL_highD_slo_distance_constraint_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60': 'MECL',
    #     # 'VICRL_highD_slo_distance_constraint_p-9e-1-1e-1_bs--1-1e3_fs-5k_nee-10_lr-5e-4_no-buffer_dm-60': 'VCIRL',
    #     "ppo_highD_no_slo_distance_dm-60": 'PPO',
    #     "ppo_lag_highD_no_slo_distance_dm-60": 'PPO_lag',
    # }

    modes = ['train']
    plot_mode = 'all'
    if plot_mode == 'part':
        for method_name in method_names_labels_dict.copy().keys():
            if 'PPO' not in method_names_labels_dict[method_name]:
                del method_names_labels_dict[method_name]
    else:
        method_names_labels_dict = method_names_labels_dict

    linestyle_all = {
        "PPO": '-' if plot_mode == 'part' else '--',
        "PPO_lag": '-' if plot_mode == 'part' else '--',
        'PPO_lag1': '-',
        'PPO_lag2': '-',
        'Bound': '-',
        "GACL": ':',  # 'GAIL',
        "BC2L": '--',  # 'Binary',
        "MECL": '-.',  # 'ICRL',
        "VCIRL-VaR": "-",
        "VCIRL-SR": "-",
        "VCIRL1": "-",
        "VCIRL2": "-",
        "VCIRL3": "-",
        "VCIRL4": "-",
        "VCIRL5": "-",
        "VCIRL-0.3": "-",
        "VCIRL-0.5": "-",
        "VCIRL-Full": "-",
        "Ram": "-",
        "Ram-1": "-",
        "Ram-0.8": "-",
        "Ram-0.5": "-",
        "Ram-0.2": "-",
        "Ram-0": "-",
        "VCIRL_Hard": "-",
    }

    linestyle_dict = {}
    for method_name in method_names_labels_dict.keys():
        for linestyle_key in linestyle_all.keys():
            if method_names_labels_dict[method_name] == linestyle_key:
                linestyle_dict.update({method_name: linestyle_all[linestyle_key]})

    for mode in modes:
        # plot_key = ['reward', 'is_collision', 'is_off_road', 'is_goal_reached', 'is_time_out']
        log_path_dict = get_plot_results_dir(env_id)

        all_mean_dict = {}
        all_std_dict = {}
        all_episodes_dict = {}
        for method_name in method_names_labels_dict.keys():
            all_results = []
            # all_valid_rewards = []
            # all_valid_episodes = []
            if method_name == 'Bound':
                results = {}
                for key in bound_results:
                    results.update({key: [bound_results[key] for item in range(max_episodes + 1000)]})
                all_results.append(results)
            else:
                for log_path in log_path_dict[method_name]:
                    monitor_path_all = []
                    if mode == 'train':
                        run_files = os.listdir(log_path)
                        for file in run_files:
                            if 'monitor' in file:
                                monitor_path_all.append(log_path + file)
                    else:
                        monitor_path_all.append(log_path + 'test/test.monitor.csv')
                    if (method_names_labels_dict[method_name] == "PPO" or
                        method_names_labels_dict[method_name] == "PPO_lag") and plot_mode != "part":
                        if 'reward_nc' in plot_key:
                            plot_key[plot_key.index('reward_nc')] = 'reward'
                    # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
                    results, valid_rewards, valid_episodes = read_running_logs(monitor_path_all=monitor_path_all,
                                                                               read_keys=plot_key,
                                                                               max_reward=max_reward,
                                                                               min_reward=min_reward,
                                                                               max_episodes=max_episodes + float(
                                                                                   max_episodes / 5),
                                                                               constraint_keys=constraint_keys)
                    if (method_names_labels_dict[method_name] == "PPO" or
                        method_names_labels_dict[method_name] == "PPO_lag") and plot_mode != "part":
                        results_copy_ = copy.copy(results)
                        for key in results.keys():
                            fill_value = np.mean(results_copy_[key][-100:])
                            results[key] = [fill_value for item in range(max_episodes + 1000)]
                    # all_valid_rewards.append(valid_rewards)
                    # all_valid_episodes.append(valid_episodes)
                    all_results.append(results)
            if mode == 'test':
                mean_std_test_results(all_results, method_name)

            mean_dict, std_dict, episodes = mean_std_plot_results(all_results)
            # mean_valid_rewards, std_valid_rewards, valid_episodes = \
            #     mean_std_plot_valid_rewards(all_valid_rewards, all_valid_episodes)
            # mean_dict.update({'reward_valid': mean_valid_rewards})
            # std_dict.update({'reward_valid': std_valid_rewards})
            # episodes.update({'reward_valid': valid_episodes})
            all_mean_dict.update({method_name: {}})
            all_std_dict.update({method_name: {}})
            all_episodes_dict.update({method_name: {}})

            if not os.path.exists(os.path.join('./plot_results/', env_id)):
                os.mkdir(os.path.join('./plot_results/', env_id))
            if not os.path.exists(os.path.join('./plot_results/', env_id, method_name)):
                os.mkdir(os.path.join('./plot_results/', env_id, method_name))

            for idx in range(len(plot_key)):
                print(method_name, plot_key[idx])
                if method_name == 'Bound' and (plot_key[idx] == 'avg_distance' or plot_key[idx] == 'avg_velocity'):
                    continue
                mean_results_moving_average = compute_moving_average(result_all=mean_dict[plot_key[idx]],
                                                                     average_num=average_num)
                std_results_moving_average = compute_moving_average(result_all=std_dict[plot_key[idx]],
                                                                    average_num=average_num)
                episode_plot = episodes[plot_key[idx]][:len(mean_results_moving_average)]
                if max_episodes:
                    mean_results_moving_average = mean_results_moving_average[:max_episodes]
                    std_results_moving_average = std_results_moving_average[:max_episodes]
                    episode_plot = episode_plot[:max_episodes]
                all_mean_dict[method_name].update({plot_key[idx]: mean_results_moving_average})
                if (method_names_labels_dict[method_name] == "PPO" or
                    method_names_labels_dict[method_name] == "PPO_lag") and plot_mode != "part":
                    all_std_dict[method_name].update({plot_key[idx]: np.zeros(std_results_moving_average.shape)})
                else:
                    all_std_dict[method_name].update({plot_key[idx]: std_results_moving_average / 2})
                all_episodes_dict[method_name].update({plot_key[idx]: episode_plot})
                plot_results(mean_results_moving_avg_dict={method_name: mean_results_moving_average},
                             std_results_moving_avg_dict={method_name: std_results_moving_average},
                             episode_plots={method_name: episode_plot},
                             label=plot_key[idx],
                             method_names=[method_name],
                             ylim=plot_y_lim_dict[plot_key[idx]],
                             save_label=os.path.join(env_id, method_name, plot_key[idx] + '_' + mode),
                             title=title,
                             axis_size=axis_size,
                             img_size=img_size,
                             linestyle_dict=linestyle_dict,
                             )
        for idx in range(len(plot_key)):
            mean_results_moving_avg_dict = {}
            std_results_moving_avg_dict = {}
            espisode_dict = {}
            plot_method_names = list(method_names_labels_dict.keys())
            for method_name in method_names_labels_dict.keys():
                if method_name == 'Bound' and (plot_key[idx] == 'avg_distance' or plot_key[idx] == 'avg_velocity'):
                    plot_method_names.remove('Bound')
                    continue
                mean_results_moving_avg_dict.update({method_name: all_mean_dict[method_name][plot_key[idx]]})
                std_results_moving_avg_dict.update({method_name: all_std_dict[method_name][plot_key[idx]]})
                espisode_dict.update({method_name: all_episodes_dict[method_name][plot_key[idx]]})
                # if (plot_key[idx] == 'reward_valid' or plot_key[idx] == 'constraint') and mode == 'test':
                #     print(method_name, plot_key[idx],
                #           all_mean_dict[method_name][plot_key[idx]][-1],
                #           all_std_dict[method_name][plot_key[idx]][-1])
                print(plot_key[idx], method_name, mean_results_moving_avg_dict[method_name][-1])
            if save:
                save_label = os.path.join(env_id, plot_key[idx] + '_' + mode + '_' + env_id + '_' + plot_mode)
            else:
                save_label = None

            plot_results(mean_results_moving_avg_dict=mean_results_moving_avg_dict,
                         std_results_moving_avg_dict=std_results_moving_avg_dict,
                         episode_plots=espisode_dict,
                         label=label_key[idx],
                         method_names=plot_method_names,
                         ylim=plot_y_lim_dict[plot_key[idx]],
                         save_label=save_label,
                         # legend_size=18,
                         legend_dict=method_names_labels_dict,
                         title=title,
                         axis_size=axis_size,
                         img_size=img_size,
                         linestyle_dict=linestyle_dict,
                         )


if __name__ == "__main__":
    generate_plots()
