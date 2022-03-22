import argparse
import os
import pickle
import shutil
import random
from collections import deque

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


def read_running_logs(log_path, read_keys):
    read_running_logs = {}

    with open(log_path, 'r') as file:
        running_logs = file.readlines()
    old_results = None

    key_indices = {}
    record_keys = running_logs[1].replace('\n', '').split(',')
    for key in read_keys:
        key_idx = record_keys.index(key)
        key_indices.update({key: key_idx})
        read_running_logs.update({key: []})

    for running_performance in running_logs[2:]:
        log_items = running_performance.split(',')
        if len(log_items) != len(record_keys):
            # continue
            results = old_results
        else:
            try:
                results = [item.replace("\n", "") for item in log_items]
                if float(results[key_indices['reward']]) > 50 or float(results[key_indices['reward']]) < -50:
                    # continue
                    results = old_results
            except:
                results = old_results
                # continue
        if results is None:
            continue
        for key in read_keys:
            read_running_logs[key].append(float(results[key_indices[key]]))
    return read_running_logs


def save_game_record(info, file, cost=None):
    is_collision = info["is_collision"]
    is_time_out = info["is_time_out"]
    is_off_road = info["is_off_road"]
    ego_velocity_x_y = info["ego_velocity"]
    ego_velocity = np.sqrt(np.sum(np.square(ego_velocity_x_y)))
    is_goal_reached = info["is_goal_reached"]
    current_step = info["current_episode_time_step"]
    if cost is None:
        file.write("{0}, {1:.3f}, {2:.0f}, {3:.0f}, {4:.0f}, {5:.0f}\n".format(current_step,
                                                                               ego_velocity,
                                                                               is_collision,
                                                                               is_off_road,
                                                                               is_goal_reached,
                                                                               is_time_out))
    else:
        file.write("{0}, {1:.3f}, {2:.3f}, {3:.0f}, {4:.0f}, {5:.0f}, {6:.0f}\n".format(current_step,
                                                                                        ego_velocity,
                                                                                        cost,
                                                                                        is_collision,
                                                                                        is_off_road,
                                                                                        is_goal_reached,
                                                                                        is_time_out))


def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss


def load_expert_data(expert_path, num_rollouts, log_file):
    file_names = os.listdir(expert_path)
    # file_names = [i for i in range(29)]
    # sample_names = random.sample(file_names, num_rollouts)
    expert_mean_reward = []
    expert_obs = []
    expert_acs = []
    expert_rs = []
    for i in range(len(file_names)):
        # file_name = sample_names[i]
        file_name = file_names[i]
        with open(os.path.join(expert_path, file_name), "rb") as f:
            data = pickle.load(f)

        data_obs = data['original_observations']
        data_acs = data['actions']
        if 'reward' in data.keys():
            data_rs = data['reward']
        else:
            data_rs = None
        total_time_step = data_acs.shape[0]
        for t in range(total_time_step - 1):
            data_obs_t = data_obs[t]
            data_obs_next_t = data_obs[t + 1]
            data_ac_t = data_acs[t]
            data_ac_next_t = data_acs[t + 1]
            if data_rs is not None:
                data_r_t = data_rs[t]
                data_r_next_t = data_rs[t + 1]
            else:
                data_r_t = 0
                data_r_next_t = 0

            expert_obs.append([data_obs_t, data_obs_next_t])
            expert_acs.append([data_ac_t, data_ac_next_t])
            expert_rs.append([data_r_t, data_r_next_t])
        # if i == 0:
        #     expert_obs = data_obs
        #     expert_acs = data_acs
        # else:
        #     expert_obs = np.concatenate([expert_obs, data_obs], axis=0)
        #     expert_acs = np.concatenate([expert_acs, data_acs], axis=0)
        expert_mean_reward.append(data['reward_sum'])
    expert_obs = np.asarray(expert_obs)
    expert_acs = np.asarray(expert_acs)
    expert_rs = np.asarray(expert_rs)
    expert_mean_reward = np.mean(expert_mean_reward)
    expert_mean_length = expert_obs.shape[0] / num_rollouts

    print('Expert_mean_reward: {0} and Expert_mean_length: {1}.'.format(expert_mean_reward, expert_mean_length),
          file=log_file,
          flush=True)

    return (expert_obs, expert_acs, expert_rs), expert_mean_reward


def load_ppo_model(model_path: str, iter_msg: str, log_file):
    if iter_msg == 'best':
        model_path = os.path.join(model_path, "best_nominal_model")
    else:
        model_path = os.path.join(model_path, 'model_{0}_itrs'.format(iter_msg), 'nominal_agent')
    print('Loading model from {0}'.format(model_path), flush=True, file=log_file)
    model = PPO.load(model_path)
    return model
