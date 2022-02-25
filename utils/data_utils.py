import argparse
import os
import shutil
from collections import deque

import yaml
import numpy as np

from gym.utils.colorize import color2num
from tqdm import tqdm

import stable_baselines3.common.callbacks as callbacks
from stable_baselines3.common.utils import safe_mean


def load_config(args=None):
    assert os.path.exists(args.config_file), "Invalid configs file {0}".format(args.config_file)
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    return config, args.DEBUG_MODE, args.LOG_FILE_PATH, args.PART_DATA


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


def read_running_logs(log_path):
    rewards = []
    is_collision = []
    is_off_road = []
    is_goal_reached = []
    is_time_out = []

    with open(log_path, 'r') as file:
        running_logs = file.readlines()
    old_results = None
    for running_performance in running_logs[2:]:
        log_items = running_performance.split(',')
        if len(log_items) != 7:
            # continue
            results = old_results
        else:
            try:
                results = [float(item.replace("\n", "")) for item in log_items]
                if results[0] > 50 or results[0] < -50:
                    # continue
                    results = old_results
            except:
                results = old_results
                # continue
        if results is None:
            continue
        rewards.append(results[0])
        is_collision.append(results[3])
        is_off_road.append(results[4])
        is_goal_reached.append(results[5])
        is_time_out.append(results[6])
        old_results = results

    return rewards, is_collision, is_off_road, is_goal_reached, is_time_out


def save_game_record(info, file):
    is_collision = info["is_collision"]
    is_time_out = info["is_time_out"]
    is_off_road = info["is_off_road"]
    is_goal_reached = info["is_goal_reached"]
    current_step = info["current_episode_time_step"]
    file.write("{0}, {1}, {2}, {3}, {4}\n".format(current_step, is_collision, is_time_out, is_off_road, is_goal_reached))
