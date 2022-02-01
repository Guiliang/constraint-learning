from utils.data_utils import read_running_logs, compute_moving_average
from utils.plot_utils import plot_curve


def plot_results(results_moving_average, label='Rewards'):
    plot_x_dict = {'PPO': [i for i in range(len(results_moving_average))]}
    plot_y_dict = {'PPO': results_moving_average}
    plot_curve(draw_keys=['PPO'],
               x_dict=plot_x_dict,
               y_dict=plot_y_dict,
               xlabel='Time Step',
               ylabel=label,
               plot_name='./plot_results/{0}'.format(label))


def generate_plots():
    plot_key = ['Rewards', 'collision_rate', 'off_road_rate', 'goal_reach_rate', 'time_out_rate']
    log_path = '../save_model/PPO-highD/train_ppo_highD-Jan-27-2022-05:04/monitor.csv'

    # rewards, is_collision, is_off_road, is_goal_reached, is_time_out = read_running_logs(log_path=log_path)
    results = read_running_logs(log_path=log_path)

    for idx in range(len(plot_key)):
        results_moving_average = compute_moving_average(result_all=results[idx], average_num=100)
        plot_results(results_moving_average, label=plot_key[idx] + '_' + log_path.split('/')[3])


if __name__ == "__main__":
    generate_plots()
