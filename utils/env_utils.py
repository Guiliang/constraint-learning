import os

import gym
import yaml

import stable_baselines3.common.vec_env as vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean, set_random_seed
from stable_baselines3.common.preprocessing import is_image_space


def make_env(env_id, env_configs, rank, log_dir, seed=0):
    def _init():
        env = gym.make(env_id, **env_configs)
        env.seed(seed + rank)
        env = Monitor(env, log_dir)
        return env

    set_random_seed(seed)
    return _init


def make_train_env(env_id, config_path, save_dir, base_seed=0, num_threads=1,
                   use_cost=False, normalize_obs=True, normalize_reward=True, normalize_cost=True,
                   **kwargs):
    with open(config_path, "r") as config_file:
        env_configs = yaml.safe_load(config_file)
    env = [make_env(env_id, env_configs, i, save_dir, base_seed)
           for i in range(num_threads)]
    env = vec_env.DummyVecEnv(env)
    if use_cost:
        env = vec_env.VecCostWrapper(env)
    if normalize_reward and normalize_cost:
        assert (all(key in kwargs for key in ['cost_info_str', 'reward_gamma', 'cost_gamma']))
        env = vec_env.VecNormalizeWithCost(
            env, training=True, norm_obs=normalize_obs, norm_reward=normalize_reward,
            norm_cost=normalize_cost, cost_info_str=kwargs['cost_info_str'],
            reward_gamma=kwargs['reward_gamma'], cost_gamma=kwargs['cost_gamma'])
    else:
        if use_cost:
            assert (all(key in kwargs for key in ['reward_gamma', 'cost_gamma']))
            env = vec_env.VecNormalizeWithCost(
                env, training=True, norm_obs=normalize_obs,
                norm_reward=normalize_reward, norm_cost=normalize_cost,
                reward_gamma=kwargs['reward_gamma'],
                cost_gamma=kwargs['cost_gamma'])
        else:
            assert (all(key in kwargs for key in ['reward_gamma']))
            env = vec_env.VecNormalize(
                env, training=True,
                norm_obs=normalize_obs, norm_reward=normalize_reward,
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


def make_eval_env(env_id, config_path, save_dir, use_cost=False, normalize_obs=True, log_file=None):
    with open(config_path, "r") as config_file:
        env_configs = yaml.safe_load(config_file)
    env_configs["test_env"] = True
    # env = [lambda: gym.make(env_id, **env_configs)]
    env = [make_env(env_id, env_configs, 0, os.path.join(save_dir, "test"))]
    env = vec_env.DummyVecEnv(env)
    if use_cost:
        env = vec_env.VecCostWrapper(env)
    print("Wrapping eval env in a VecNormalize.", file=log_file, flush=True)
    if use_cost:
        env = vec_env.VecNormalizeWithCost(env, training=False, norm_obs=normalize_obs,
                                           norm_reward=False, norm_cost=False)
    else:
        env = vec_env.VecNormalize(env, training=False, norm_obs=normalize_obs, norm_reward=False)

    if is_image_space(env.observation_space) and not isinstance(env, vec_env.VecTransposeImage):
        print("Wrapping eval env in a VecTransposeImage.")
        env = vec_env.VecTransposeImage(env)

    return env
