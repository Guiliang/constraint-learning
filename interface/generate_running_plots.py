import os

from utils.data_utils import read_running_logs, compute_moving_average
from utils.plot_utils import plot_curve


def plot_results(results_moving_average, label='Rewards', save_label=''):
    plot_x_dict = {'PPO': [i for i in range(len(results_moving_average))]}
    plot_y_dict = {'PPO': results_moving_average}
    plot_curve(draw_keys=['PPO'],
               x_dict=plot_x_dict,
               y_dict=plot_y_dict,
               xlabel='Episode',
               ylabel=label,
               title='{0}'.format(save_label),
               plot_name='./plot_results/{0}'.format(save_label))


def generate_plots():
    modes = ['train', 'test']

    for mode in modes:
        plot_key = ['Rewards', 'collision_rate', 'off_road_rate', 'goal_reach_rate', 'time_out_rate']
        # log_path = '../save_model/ICRL-highD/part-train_ICRL_highD_collision_constraint-Mar-07-2022-07:03/'
        # log_path = '../save_model/ICRL-highD/part-train_ICRL_highD_offroad_constraint-Mar-07-2022-07:01/'
        log_path = '../save_model/ICRL-highD/train_ICRL_highD_offroad_constraint-Mar-13-2022-23:37/'
        # log_path = '../save_model/ICRL-highD/train_ICRL_highD_collision_constraint-multi_env-Mar-15-2022-07:27/'

        # log_path = '../save_model/PPO-highD/train_ppo_highD-multi_env-Mar-10-2022-04:37/'
        # log_path = '../save_model/PPO-highD/train_ppo_highD_no_collision-multi_env-Mar-18-2022-02:32-seed_123/'
        # log_path = '../save_model/PPO-highD/train_ppo_highD_no_collision-multi_env-Mar-16-2022-00:14-seed_321/'
        # log_path = '../save_model/PPO-highD/train_ppo_highD_no_offroad-multi_env-Mar-10-2022-04:38/'
        # log_path = '../save_model/PPO-highD/train_ppo_highD_no_collision-multi_env-Mar-10-2022-00:18/'
        # log_path = '../save_model/PPO-highD/part-train_ppo_highD-Feb-01-2022-10:31/'
        # log_path = '../save_model/PPO-highD/part-train_ppo_highD_no_collision-Mar-03-2022-11:04/'
        # log_path = '../save_model/PPO-highD/part-train_ppo_highD_no_offroad-Mar-03-2022-11:05/'
        if mode == 'train':
            log_path += 'monitor.csv'
        else:
            log_path += 'test/test.monitor.csv'

        # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
        results = read_running_logs(log_path=log_path)

        if not os.path.exists('./plot_results/'+log_path.split('/')[3]):
            os.mkdir('./plot_results/'+log_path.split('/')[3])

        for idx in range(len(plot_key)):
            results_moving_average = compute_moving_average(result_all=results[idx], average_num=100)
            plot_results(results_moving_average,
                         label=plot_key[idx],
                         save_label=log_path.split('/')[3] + '/'+plot_key[idx] + '_' + mode)


if __name__ == "__main__":
    generate_plots()
