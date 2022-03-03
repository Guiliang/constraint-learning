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
    mode = 'train'
    plot_key = ['Rewards', 'collision_rate', 'off_road_rate', 'goal_reach_rate', 'time_out_rate']
    # log_path = '../save_model/PPO-highD/train_ppo_highD-Feb-01-2022-10:31/'
    # log_path = '../save_model/PPO-highD/train_ppo_highD_percent_0_5-Feb-01-2022-10:28/'
    # log_path = '../save_model/PPO-highD/train_ppo_highD-Jan-27-2022-05:04/'
    # log_path = '../save_model/PPO-highD/train_ppo_highD_no_collision-Feb-11-2022-08:57/'
    # log_path = '../save_model/PPO-highD/train_ppo_highD_no_offroad-Feb-11-2022-08:58/'
    # log_path = '../save_model/ICRL-highD/part-train_ICRL_highD_offroad_constraint-Feb-25-2022-05:53/'
    # log_path = '../save_model/ICRL-highD/part-train_ICRL_highD_collision_constraint-Feb-25-2022-05:47/'
    log_path = '../save_model/PPO-highD/part-train_ppo_highD_no_collision-Mar-02-2022-00:38/'
    if mode == 'train':
        log_path += 'monitor.csv'
    else:
        log_path += 'test/test.monitor.csv'

    # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
    results = read_running_logs(log_path=log_path)

    for idx in range(len(plot_key)):
        results_moving_average = compute_moving_average(result_all=results[idx],
                                                        average_num=100)
        plot_results(results_moving_average,
                     label=plot_key[idx],
                     save_label=plot_key[idx] + '_' + log_path.split('/')[3] + '_' + mode)


if __name__ == "__main__":
    generate_plots()
