import os
import numpy as np
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


def make_eval_env(env_id, config_path, save_dir, mode='test', use_cost=False, normalize_obs=True, log_file=None):
    with open(config_path, "r") as config_file:
        env_configs = yaml.safe_load(config_file)
    env_configs["test_env"] = True
    # env = [lambda: gym.make(env_id, **env_configs)]
    env = [make_env(env_id, env_configs, 0, os.path.join(save_dir, mode))]
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


def sample_from_agent(agent, env, rollouts):
    if isinstance(env, vec_env.VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    orig_observations, observations, actions = [], [], []
    rewards, lengths = [], []
    for i in range(rollouts):
        # Avoid double reset, as VecEnv are reset automatically
        if i == 0:
            obs = env.reset()
        # benchmark_id = env.venv.envs[0].benchmark_id
        # print('senario id', benchmark_id)

        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            orig_observations.append(env.get_original_obs())
            observations.append(obs)

            action, state = agent.predict(obs, state=state, deterministic=False)
            obs, reward, done, _info = env.step(action)

            actions.append(action)
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)

    orig_observations = np.squeeze(np.array(orig_observations), axis=1)
    observations = np.squeeze(np.array(observations), axis=1)
    actions = np.squeeze(np.array(actions), axis=1)
    rewards = np.squeeze(np.array(rewards), axis=1)
    lengths = np.array(lengths)

    return orig_observations, observations, actions, rewards, lengths