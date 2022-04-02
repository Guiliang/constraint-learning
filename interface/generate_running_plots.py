import os

from utils.data_utils import read_running_logs, compute_moving_average, average_plot_results
from utils.plot_utils import plot_curve


def plot_results(results_moving_average, ylim, label, method_name, save_label):
    plot_x_dict = {method_name: [i for i in range(len(results_moving_average))]}
    plot_y_dict = {method_name: results_moving_average}
    plot_curve(draw_keys=[method_name],
               x_dict=plot_x_dict,
               y_dict=plot_y_dict,
               ylim=ylim,
               xlabel='Episode',
               ylabel=label,
               title='{0}'.format(save_label),
               plot_name='./plot_results/{0}'.format(save_label))


def generate_plots():
    modes = ['train', 'test']

    for mode in modes:
        # plot_key = ['reward', 'is_collision', 'is_off_road', 'is_goal_reached', 'is_time_out']

        plot_key = ['reward', 'is_collision', 'is_off_road', 'is_goal_reached', 'is_time_out', 'avg_velocity',
                    'is_over_speed']
        plot_y_lim_dict = {'reward': (-40, 40),
                           'is_collision': (0, 1),
                           'is_off_road': (0, 1),
                           'is_goal_reached': (0, 1),
                           'is_time_out': (0, 1),
                           'avg_velocity': (20, 50),
                           'is_over_speed': (0, 1)}

        file_type = "PPO_highD_velocity"

        log_path_dict = {
            "PPO_highD_velocity": [
                '../save_model/PPO-highD/train_ppo_highD_velocity_penalty-multi_env-Mar-20-2022-10:21-seed_123/',
                '../save_model/PPO-highD/train_ppo_highD_velocity_penalty-multi_env-Mar-21-2022-05:29-seed_123/',
            ],
            "PPO_highD_no-velocity": [
                '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty-multi_env-Mar-20-2022-10:18-seed_123/',
                '../save_model/PPO-highD/train_ppo_highD_no_velocity_penalty-multi_env-Mar-21-2022-05:30-seed_123/'
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
            "VICRL_highD_velocity-dim2-buff": [
                '../save_model/VICRL-highD/train_VICRL_highD_velocity_constraint_no_is_p-1-1_dim-2-multi_env-Mar-31-2022-06:36-seed_123/'
            ],
        }
        all_results = []
        for log_path in log_path_dict[file_type]:
            if mode == 'train':
                log_path += 'monitor.csv'
            else:
                log_path += 'test/test.monitor.csv'

            # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
            results = read_running_logs(log_path=log_path, read_keys=plot_key)
            all_results.append(results)

        avg_results = average_plot_results(all_results)

        # if not os.path.exists('./plot_results/' + log_path.split('/')[3]):
        #     os.mkdir('./plot_results/' + log_path.split('/')[3])
        if not os.path.exists('./plot_results/' + file_type):
            os.mkdir('./plot_results/' + file_type)

        for idx in range(len(plot_key)):
            results_moving_average = compute_moving_average(result_all=avg_results[plot_key[idx]], average_num=100)
            plot_results(results_moving_average,
                         label=plot_key[idx],
                         method_name=file_type,
                         ylim=plot_y_lim_dict[plot_key[idx]],
                         save_label=file_type + '/' + plot_key[idx] + '_' + mode)


if __name__ == "__main__":
    generate_plots()
