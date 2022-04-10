import os
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gym
import yaml
import stable_baselines3.common.vec_env as vec_env
from common.cns_monitor import CNSMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean, set_random_seed
from stable_baselines3.common.preprocessing import is_image_space


def get_benchmark_ids(num_threads, benchmark_idx, benchmark_total_nums, env_ids):
    benchmark_ids = []
    for i in range(num_threads):
        if benchmark_total_nums[i] > benchmark_idx:
            benchmark_ids.append(env_ids[i][benchmark_idx])
        else:
            benchmark_ids.append(None)
    return benchmark_ids


def multi_threads_sample_from_agent(agent, env, rollouts, num_threads, store_by_game=False):
    # if isinstance(env, vec_env.VecEnv):
    #     assert env.num_envs == 1, "You must pass only one environment when using this function"
    rollouts = int(float(rollouts) / num_threads)
    all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
    sum_rewards, all_lengths = [], []
    max_benchmark_num, env_ids, benchmark_total_nums = get_all_env_ids(num_threads, env)
    assert rollouts <= min(benchmark_total_nums)
    for j in range(rollouts):
        benchmark_ids = get_benchmark_ids(num_threads=num_threads, benchmark_idx=j,
                                          benchmark_total_nums=benchmark_total_nums, env_ids=env_ids)
        obs = env.reset_benchmark(benchmark_ids=benchmark_ids)  # force reset for all games
        multi_thread_already_dones = [False for i in range(num_threads)]
        done, states = False, None
        episode_sum_rewards = [0 for i in range(num_threads)]
        episode_lengths = [0 for i in range(num_threads)]
        origin_obs_game = [[] for i in range(num_threads)]
        obs_game = [[] for i in range(num_threads)]
        acs_game = [[] for i in range(num_threads)]
        rs_game = [[] for i in range(num_threads)]
        while not done:
            actions, states = agent.predict(obs, state=states, deterministic=False)
            original_obs = env.get_original_obs()
            new_obs, rewards, dones, _infos = env.step(actions)
            # benchmark_ids = [env.venv.envs[i].benchmark_id for i in range(num_threads)]
            # print(benchmark_ids)
            for i in range(num_threads):
                if not multi_thread_already_dones[i]:  # we will not store when a game is done
                    origin_obs_game[i].append(original_obs[i])
                    obs_game[i].append(obs[i])
                    acs_game[i].append(actions[i])
                    rs_game[i].append(rewards[i])
                    episode_sum_rewards[i] += rewards[i]
                    episode_lengths[i] += 1
                if dones[i]:
                    multi_thread_already_dones[i] = True
            done = True
            for multi_thread_done in multi_thread_already_dones:
                if not multi_thread_done:  # we will wait for all games done
                    done = False
                    break
            obs = new_obs
        origin_obs_game = [np.array(origin_obs_game[i]) for i in range(num_threads)]
        obs_game = [np.array(obs_game[i]) for i in range(num_threads)]
        acs_game = [np.array(acs_game[i]) for i in range(num_threads)]
        rs_game = [np.array(rs_game[i]) for i in range(num_threads)]
        all_orig_obs += origin_obs_game
        all_obs += obs_game
        all_acs += acs_game
        all_rs += rs_game

        sum_rewards += episode_sum_rewards
        all_lengths += episode_lengths
    if not store_by_game:
        all_orig_obs = np.concatenate(all_orig_obs, axis=0)
        all_obs = np.concatenate(all_obs, axis=0)
        all_acs = np.concatenate(all_acs, axis=0)
        all_rs = np.concatenate(all_rs, axis=0)
    return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, all_lengths


def sample_from_agent(agent, env, rollouts, store_by_game=False):
    if isinstance(env, vec_env.VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
    sum_rewards, lengths = [], []
    for i in range(rollouts):
        # Avoid double reset, as VecEnv are reset automatically
        if i == 0:
            obs = env.reset()
        # benchmark_id = env.venv.envs[0].benchmark_id
        # print('senario id', benchmark_id)

        done, state = False, None
        episode_sum_reward = 0.0
        episode_length = 0
        if store_by_game:
            origin_obs_game = []
            obs_game = []
            acs_game = []
            rs_game = []
        while not done:
            action, state = agent.predict(obs, state=state, deterministic=False)

            if store_by_game:
                origin_obs_game.append(env.get_original_obs())
                obs_game.append(obs)
                acs_game.append(action)
            else:
                all_orig_obs.append(env.get_original_obs())
                all_obs.append(obs)
                all_acs.append(action)
            obs, reward, done, _info = env.step(action)
            if store_by_game:
                rs_game.append(reward)
            else:
                all_rs.append(reward)

            episode_sum_reward += reward
            episode_length += 1
        if store_by_game:
            origin_obs_game = np.squeeze(np.array(origin_obs_game), axis=1)
            obs_game = np.squeeze(np.array(obs_game), axis=1)
            acs_game = np.squeeze(np.array(acs_game), axis=1)
            rs_game = np.squeeze(np.asarray(rs_game))
            all_orig_obs.append(origin_obs_game)
            all_obs.append(obs_game)
            all_acs.append(acs_game)
            all_rs.append(rs_game)

        sum_rewards.append(episode_sum_reward)
        lengths.append(episode_length)

    if store_by_game:
        return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths
    else:
        all_orig_obs = np.squeeze(np.array(all_orig_obs), axis=1)
        all_obs = np.squeeze(np.array(all_obs), axis=1)
        all_acs = np.squeeze(np.array(all_acs), axis=1)
        all_rs = np.array(all_rs)
        sum_rewards = np.squeeze(np.array(sum_rewards), axis=1)
        lengths = np.array(lengths)
        return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths


class ExternalRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Wrapper, group: str, wrapper_config: dict):
        super(ExternalRewardWrapper, self).__init__(env=env)
        self.wrapper_config = wrapper_config
        self.group = group

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        observation, reward, done, info = self.env.step(action)

        reward_features = self.wrapper_config['reward_features']
        feature_bounds = self.wrapper_config['feature_bounds']
        feature_penalties = self.wrapper_config['feature_penalties']
        terminates = self.wrapper_config['terminate']
        lag_cost = 0
        for idx in range(len(reward_features)):
            reward_feature = reward_features[idx]
            if reward_feature == 'velocity':
                ego_velocity_x_y = info["ego_velocity"]
                ego_velocity = np.sqrt(np.sum(np.square(ego_velocity_x_y)))
                if ego_velocity > float(feature_bounds[idx][1]):
                    reward += float(feature_penalties[idx])
                    lag_cost = 1
                    if terminates[idx]:
                        done = True
                    info.update({'is_over_speed': 1})
                else:
                    info.update({'is_over_speed': 0})
            else:
                raise ValueError("Unknown reward features: {0}".format(reward_feature))
        # print(ego_velocity, lag_cost)
        info.update({'lag_cost': lag_cost})
        return observation, reward, done, info


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        save_path_name = os.path.join(self.save_path, "vecnormalize.pkl")
        self.model.get_vec_normalize_env().save(save_path_name)
        print("Saved vectorized and normalized environment to {}".format(save_path_name))


def get_all_env_ids(num_threads, env):
    max_benchmark_num = 0
    env_ids = []
    benchmark_total_nums = []
    for i in range(num_threads):
        try:  # we need to change this setting if you modify the number of env wrappers.
            env_ids.append(list(env.venv.envs[i].env.env.env.all_problem_dict.keys()))
        except:
            env_ids.append(list(env.venv.envs[i].env.env.all_problem_dict.keys()))
        benchmark_total_nums.append(len(env_ids[i]))
        if len(env_ids[i]) > max_benchmark_num:
            max_benchmark_num = len(env_ids[i])
    return max_benchmark_num, env_ids, benchmark_total_nums
