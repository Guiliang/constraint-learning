import logging
import os
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gym
import yaml
import stable_baselines3.common.vec_env as vec_env
from common.cns_monitor import CNSMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean, set_random_seed
from stable_baselines3.common.preprocessing import is_image_space


def make_env(env_id, env_configs, rank, log_dir, multi_env=False, seed=0):
    def _init():
        if 'commonroad' in env_id:
            # import commonroad_environment.commonroad_rl.gym_commonroad
            from commonroad_environment.commonroad_rl import gym_commonroad
        elif 'HC' in env_id:
            # from mujuco_environment.custom_envs.envs import half_cheetah
            import mujuco_environment.custom_envs
        env_configs_copy = copy(env_configs)
        if multi_env and 'commonroad' in env_id:
            env_configs_copy.update({'train_reset_config_path': env_configs['train_reset_config_path'] + '/{0}'.format(rank)}),
        if 'external_reward' in env_configs:
            del env_configs_copy['external_reward']
        env = gym.make(id=env_id,
                       **env_configs_copy)
        env.seed(seed + rank)
        del env_configs_copy
        if 'external_reward' in env_configs:
            print("Using external reward", flush=True)
            env = ExternalRewardWrapper(env=env, wrapper_config=env_configs['external_reward'])
        monitor_rank = None
        if multi_env:
            monitor_rank = rank
        env = CNSMonitor(env=env, filename=log_dir, rank=monitor_rank)
        return env

    set_random_seed(seed)
    return _init


def make_train_env(env_id, config_path, save_dir, group='PPO', base_seed=0, num_threads=1,
                   use_cost=False, normalize_obs=True, normalize_reward=True, normalize_cost=True, multi_env=False,
                   log_file=None, part_data=False,
                   **kwargs):
    if config_path is not None:
        with open(config_path, "r") as config_file:
            env_configs = yaml.safe_load(config_file)
            if multi_env:
                env_configs['train_reset_config_path'] += '_split'
            if part_data:
                env_configs['train_reset_config_path'] += '_debug'
                env_configs['test_reset_config_path'] += '_debug'
                env_configs['meta_scenario_path'] += '_debug'
    else:
        env_configs = {}
    env = [make_env(env_id=env_id,
                    env_configs=env_configs,
                    rank=i,
                    log_dir=save_dir,
                    multi_env=multi_env,
                    seed=base_seed)
           for i in range(num_threads)]
    if 'HC' in env_id:
        env = vec_env.SubprocVecEnv(env)
    elif 'commonroad' in env_id:
        env = vec_env.DummyVecEnv(env)
    else:
        raise ValueError("Unknown env id {0}".format(env_id))
    # print("The obs space is {0}".format(len(env.observation_space.high)), file=log_file, flush=True)
    if use_cost:
        env = vec_env.VecCostWrapper(env)
    if normalize_reward and normalize_cost:
        assert (all(key in kwargs for key in ['cost_info_str', 'reward_gamma', 'cost_gamma']))
        env = vec_env.VecNormalizeWithCost(
            env, training=True,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            norm_cost=normalize_cost,
            cost_info_str=kwargs['cost_info_str'],
            reward_gamma=kwargs['reward_gamma'],
            cost_gamma=kwargs['cost_gamma'])
    else:
        if 'ICRL' in group:
            assert (all(key in kwargs for key in ['reward_gamma', 'cost_gamma']))
            env = vec_env.VecNormalizeWithCost(
                env, training=True,
                norm_obs=normalize_obs,
                norm_reward=normalize_reward,
                norm_cost=normalize_cost,
                reward_gamma=kwargs['reward_gamma'],
                cost_gamma=kwargs['cost_gamma'])
        else:
            assert (all(key in kwargs for key in ['reward_gamma']))
            env = vec_env.VecNormalize(
                env,
                training=True,
                norm_obs=normalize_obs,
                norm_reward=normalize_reward,
                gamma=kwargs['reward_gamma'])
    # else:
    #     if use_cost:
    #         env = vec_env.VecNormalizeWithCost(
    #             env, training=True, norm_obs=normalize_obs, norm_reward=False, norm_cost=False)
    #     else:
    #         env = vec_env.VecNormalize(
    #             env, training=True, norm_obs=normalize_obs, norm_reward=False, norm_cost=False,
    #             gamma=kwargs['reward_gamma'])
    return env


def make_eval_env(env_id, config_path, save_dir, group='PPO', mode='test', use_cost=False, normalize_obs=True,
                  part_data=False, log_file=None):

    if config_path is not None:
        with open(config_path, "r") as config_file:
            env_configs = yaml.safe_load(config_file)
            if part_data:
                env_configs['train_reset_config_path'] += '_debug'
                env_configs['test_reset_config_path'] += '_debug'
                env_configs['meta_scenario_path'] += '_debug'
        env_configs["test_env"] = True
    else:
        env_configs = {}
    # env = [lambda: gym.make(env_id, **env_configs)]
    env = [make_env(env_id, env_configs, 0, os.path.join(save_dir, mode))]
    if 'HC' in env_id:
        env = vec_env.SubprocVecEnv(env)
    elif 'commonroad' in env_id:
        env = vec_env.DummyVecEnv(env)
    else:
        raise ValueError("Unknown env id {0}".format(env_id))
    if use_cost:
        env = vec_env.VecCostWrapper(env)
    print("Wrapping eval env in a VecNormalize.", file=log_file, flush=True)
    if 'ICRL' in group:
        env = vec_env.VecNormalizeWithCost(env, training=False, norm_obs=normalize_obs,
                                           norm_reward=False, norm_cost=False)
    else:
        env = vec_env.VecNormalize(env, training=False, norm_obs=normalize_obs, norm_reward=False)

    if is_image_space(env.observation_space) and not isinstance(env, vec_env.VecTransposeImage):
        print("Wrapping eval env in a VecTransposeImage.")
        env = vec_env.VecTransposeImage(env)

    return env


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
    def __init__(self, env: gym.Wrapper, wrapper_config):
        super(ExternalRewardWrapper, self).__init__(env=env)
        self.wrapper_config = wrapper_config

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        observation, reward, done, info = self.env.step(action)

        reward_features = self.wrapper_config['reward_features']
        feature_bounds = self.wrapper_config['feature_bounds']
        feature_penalties = self.wrapper_config['feature_penalties']
        terminates = self.wrapper_config['terminate']
        for idx in range(len(reward_features)):
            reward_feature = reward_features[idx]
            if reward_feature == 'velocity':
                ego_velocity_x_y = info["ego_velocity"]
                ego_velocity = np.sqrt(np.sum(np.square(ego_velocity_x_y)))
                if ego_velocity > float(feature_bounds[idx][1]):
                    reward += float(feature_penalties[idx])
                    info.update({'is_over_speed': 1})
                    if terminates[idx]:
                        done = True
                else:
                    info.update({'is_over_speed': 0})
            else:
                raise ValueError("Unknown reward features: {0}".format(reward_feature))

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