import argparse
import os
import pickle
import shutil
from collections import deque
import time
import pickle5
import psutil
import yaml
import numpy as np
from gym.utils.colorize import color2num
from tqdm import tqdm
import stable_baselines3.common.callbacks as callbacks
from stable_baselines3 import PPO
from stable_baselines3.common.utils import safe_mean


def load_config(args=None):
    assert os.path.exists(args.config_file), "Invalid configs file {0}".format(args.config_file)
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    return config, args.DEBUG_MODE, args.LOG_FILE_PATH, args.PART_DATA, int(args.NUM_THREADS), int(args.SEED)


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to configs file")
    # parser.add_argument("-t", "--train_flag", help="if training",
    #                     dest="TRAIN_FLAG",
    #                     default='1', required=False)
    parser.add_argument("-d", "--debug_mode", help="whether to use the debug mode",
                        dest="DEBUG_MODE",
                        default=False, required=False)
    parser.add_argument("-p", "--part_data", help="whether to use the partial dataset",
                        dest="PART_DATA",
                        default=False, required=False)
    parser.add_argument("-n", "--num_threads", help="number of threads for loading envs.",
                        dest="NUM_THREADS",
                        default=1, required=False)
    parser.add_argument("-s", "--seed", help="the seed of randomness",
                        dest="SEED",
                        default=123,
                        required=False,
                        type=int)
    parser.add_argument("-l", "--log_file", help="log file", dest="LOG_FILE_PATH", default=None, required=False)
    args = parser.parse_args()
    return args


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


# This callback should be used with the 'with' block, to allow for correct
# initialisation and destruction
class ProgressBarManager:
    def __init__(self, total_timesteps):  # init object with total timesteps
        self.pbar = None
        self.total_timesteps = int(total_timesteps)

    def __enter__(self):  # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps, dynamic_ncols=True)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()


# =============================================================================
# Custom callbacks
# =============================================================================

class ProgressBarCallback(callbacks.BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """

    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = int(self.num_timesteps)
        self._pbar.update(0)

    def _on_rollout_end(self):
        total_reward = safe_mean([ep_info["reward"] for ep_info in self.model.ep_info_buffer])
        try:
            average_cost = safe_mean(self.model.rollout_buffer.orig_costs)
            total_cost = np.sum(self.model.rollout_buffer.orig_costs)
            self._pbar.set_postfix(
                tr='%05.1f' % total_reward,
                ac='%05.3f' % average_cost,
                tc='%05.1f' % total_cost,
                nu='%05.1f' % self.model.dual.nu().item()
            )
        except:  # No cost
            # average_cost = 0
            # total_cost = 0
            # desc = "No Cost !!!"
            self._pbar.set_postfix(
                tr='%05.1f' % total_reward,
                ac='No Cost',
                tc='No Cost',
                nu='No Dual'
            )


def del_and_make(d):
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)


def compute_moving_average(result_all, average_num=100):
    result_moving_average_all = []
    moving_values = deque([], maxlen=average_num)
    for result in result_all:
        moving_values.append(result)
        result_moving_average_all.append(np.mean(moving_values))
    return result_moving_average_all


def read_running_logs(monitor_path_all, read_keys,max_reward,min_reward):
    read_running_logs = {}

    # handle the keys
    with open(monitor_path_all[0], 'r') as file:
        running_logs = file.readlines()
    key_indices = {}
    record_keys = running_logs[1].replace('\n', '').split(',')
    if len(record_keys) > 10:
        raise ValueError("Something wrong with the file {0}".format(monitor_path_all[0]))
    for key in read_keys:
        key_idx = record_keys.index(key)
        key_indices.update({key: key_idx})
        read_running_logs.update({key: []})

    # read all the logs
    running_logs_all = []
    max_len = 0
    for monitor_path in monitor_path_all:
        with open(monitor_path, 'r') as file:
            running_logs = file.readlines()
        running_logs_all.append(running_logs[2:])
        if len(running_logs[2:]) > max_len:
            max_len = len(running_logs[2:])

    # iteratively read the logs
    line_num = 0
    while line_num < max_len:
        old_results = None
        for i in range(len(monitor_path_all)):
            if line_num >= len(running_logs_all[i]):
                continue
            running_performance = running_logs_all[i][line_num]
            log_items = running_performance.split(',')
            if len(log_items) != len(record_keys):
                # continue
                results = old_results
            else:
                try:
                    results = [item.replace("\n", "") for item in log_items]
                    if float(results[key_indices['reward']]) > max_reward or float(results[key_indices['reward']]) < min_reward:
                        # continue
                        results = old_results
                except:
                    results = old_results
                    # continue
            if results is None:
                continue
            for key in read_keys:
                read_running_logs[key].append(float(results[key_indices[key]]))
        line_num += 1

    return read_running_logs


def save_game_record(info, file, cost=None):
    is_collision = info["is_collision"]
    is_time_out = info["is_time_out"]
    is_off_road = info["is_off_road"]
    ego_velocity_x_y = info["ego_velocity"]
    # ego_velocity = np.sqrt(np.sum(np.square(ego_velocity_x_y)))
    ego_velocity_x = ego_velocity_x_y[0]
    ego_velocity_y = ego_velocity_x_y[1]
    is_goal_reached = info["is_goal_reached"]
    current_step = info["current_episode_time_step"]
    if cost is None:
        file.write("{0}, {1:.3f}, {2:.3f}, {3:.0f}, {4:.0f}, {5:.0f}, {6:.0f}\n".format(current_step,
                                                                                        ego_velocity_x,
                                                                                        ego_velocity_y,
                                                                                        is_collision,
                                                                                        is_off_road,
                                                                                        is_goal_reached,
                                                                                        is_time_out))
    else:
        file.write("{0}, {1:.3f}, {2:.3f}, {3:.3f}, {4:.0f}, {5:.0f}, {6:.0f}, {7:.0f}\n".format(current_step,
                                                                                                 ego_velocity_x,
                                                                                                 ego_velocity_y,
                                                                                                 cost,
                                                                                                 is_collision,
                                                                                                 is_off_road,
                                                                                                 is_goal_reached,
                                                                                                 is_time_out))


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def bak_load_expert_data(expert_path, num_rollouts):
    expert_mean_reward = []
    for i in range(num_rollouts):
        with open(os.path.join(expert_path, "files/EXPERT/rollouts", "%s.pkl" % str(i)), "rb") as f:
            data = pickle.load(f)

        if i == 0:
            expert_obs = data['observations']
            expert_acs = data['actions']
        else:
            expert_obs = np.concatenate([expert_obs, data['observations']], axis=0)
            expert_acs = np.concatenate([expert_acs, data['actions']], axis=0)

        expert_mean_reward.append(data['rewards'])

    expert_mean_reward = np.mean(expert_mean_reward)
    expert_mean_length = expert_obs.shape[0] / num_rollouts

    return (expert_obs, expert_acs), expert_mean_reward


def load_expert_data(expert_path,
                     num_rollouts=None,
                     use_pickle5=False,
                     store_by_game=False,
                     add_next_step=True,
                     log_file=None):
    file_names = sorted(os.listdir(expert_path))
    # file_names = [i for i in range(29)]
    # sample_names = random.sample(file_names, num_rollouts)
    expert_mean_reward = []
    expert_obs = []
    expert_acs = []
    expert_rs = []
    num_samples = 0
    if num_rollouts is None:
        num_rollouts = len(file_names)
    for i in range(num_rollouts):
        # file_name = sample_names[i]
        file_name = file_names[i]
        with open(os.path.join(expert_path, file_name), "rb") as f:
            if use_pickle5:  # the data is stored by pickle5
                data = pickle5.load(f)
            else:
                data = pickle.load(f)
        if use_pickle5:  # for the mujoco data, observations are the original_observations
            data_obs = data['observations']
        else:
            data_obs = data['original_observations']
        data_acs = data['actions']
        if 'reward' in data.keys():
            data_rs = data['reward']
        else:
            data_rs = None
        if add_next_step:
            total_time_step = data_acs.shape[0] - 1
        else:
            total_time_step = data_acs.shape[0]

        if store_by_game:
            expert_obs_game = []
            expert_acs_game = []
            expert_rs_game = []

        for t in range(total_time_step):
            data_obs_t = data_obs[t]
            data_ac_t = data_acs[t]
            if add_next_step:
                data_obs_next_t = data_obs[t + 1]
                data_ac_next_t = data_acs[t + 1]
            num_samples += 1
            if data_rs is not None:
                data_r_t = data_rs[t]
                if add_next_step:
                    data_r_next_t = data_rs[t + 1]
            else:
                data_r_t = 0
                if add_next_step:
                    data_r_next_t = 0
            if add_next_step:
                data_obs_t_store = [data_obs_t, data_obs_next_t]
                data_acs_t_store = [data_ac_t, data_ac_next_t]
                data_r_t_store = [data_r_t, data_r_next_t]
            else:
                data_obs_t_store = data_obs_t
                data_acs_t_store = data_ac_t
                data_r_t_store = data_r_t
            if store_by_game:
                expert_obs_game.append(data_obs_t_store)
                expert_acs_game.append(data_acs_t_store)
                expert_rs_game.append(data_r_t_store)
            else:
                expert_obs.append(data_obs_t_store)
                expert_acs.append(data_acs_t_store)
                expert_rs.append(data_r_t_store)

        if store_by_game:
            expert_obs.append(np.asarray(expert_obs_game))
            expert_acs.append(np.asarray(expert_acs_game))
            expert_rs.append(np.asarray(expert_rs_game))
        if use_pickle5:  # for the mujoco data, rewards are the reward_sums
            expert_mean_reward.append(data['rewards'])
        else:
            expert_mean_reward.append(data['reward_sum'])
    expert_mean_reward = np.mean(expert_mean_reward)
    expert_mean_length = num_samples / len(file_names)
    print('Expert_mean_reward: {0} and Expert_mean_length: {1}.'.format(expert_mean_reward, expert_mean_length),
          file=log_file,
          flush=True)
    if store_by_game:
        return (expert_obs, expert_acs, expert_rs), expert_mean_reward
    else:
        expert_obs = np.asarray(expert_obs)
        expert_acs = np.asarray(expert_acs)
        expert_rs = np.asarray(expert_rs)
        return (expert_obs, expert_acs, expert_rs), expert_mean_reward


def load_ppo_model(model_path: str, iter_msg: str, log_file):
    if iter_msg == 'best':
        model_path = os.path.join(model_path, "best_nominal_model")
    else:
        model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
    print('Loading model from {0}'.format(model_path), flush=True, file=log_file)
    model = PPO.load(model_path)
    return model


def get_benchmark_ids(num_threads, benchmark_idx, benchmark_total_nums, env_ids):
    benchmark_ids = []
    for i in range(num_threads):
        if benchmark_total_nums[i] > benchmark_idx:
            benchmark_ids.append(env_ids[i][benchmark_idx])
        else:
            benchmark_ids.append(None)
    return benchmark_ids


def get_obs_feature_names(env, env_id):
    feature_names = []
    if 'commonroad' in env_id:
        try:  # we need to change this setting if you modify the number of env wrappers.
            observation_space_dict = env.venv.envs[0].env.env.env.observation_collector.observation_space_dict
        except:
            observation_space_dict = env.venv.envs[0].env.env.observation_collector.observation_space_dict
        observation_space_names = observation_space_dict.keys()
        for key in observation_space_names:
            feature_len = observation_space_dict[key].shape[0]
            for i in range(feature_len):
                feature_names.append(key + '_' + str(i))
    if 'HC' in env_id:
        feature_names.append('(pls visit: {0})'.format(
            'https://github.com/openai/gym/blob/master/gym/envs/mujoco/assets/half_cheetah.xml'))
    return feature_names


def get_input_features_dim(feature_select_names, all_feature_names):
    if feature_select_names is None:
        feature_select_dim = None
    else:
        feature_select_dim = []
        for feature_name in feature_select_names:
            if feature_name == -1:
                feature_select_dim.append(-1)  # -1 indicates don't select
                break
            else:
                feature_select_dim.append(all_feature_names.index(feature_name))
    return feature_select_dim


def average_plot_results(all_results):
    results_avg = {}
    for key in all_results[0]:
        all_plot_values = []
        max_len = 0
        for results in all_results:
            plot_values = results[key]
            if len(plot_values) > max_len:
                max_len = len(plot_values)
            all_plot_values.append(plot_values)
        # max_len = 1000
        avg_plot_values = []
        for i in range(max_len):
            plot_value_t = []
            for plot_values in all_plot_values:
                if len(plot_values) > i:
                    plot_value_t.append(plot_values[i])
            avg_plot_values.append(np.mean(plot_value_t))
        results_avg.update({key: avg_plot_values})

    return results_avg


def print_resource(mem_prev, time_prev, process_name, log_file):
    mem_current = process_memory()
    time_current = time.time()
    print("{0} consumed memory: {1:.2f}/{2:.2f} and time {3:.2f}".format(
        process_name,
        float(mem_current - mem_prev) / 1000000,
        float(mem_current) / 1000000,
        time_current - time_prev), file=log_file, flush=True)
    return mem_current, time_current
