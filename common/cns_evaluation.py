from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


def evaluate_icrl_policy(
        model: "base_class.BaseAlgorithm",
        env: Union[gym.Env, VecEnv],
        record_info_names: list,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        callback: Optional[Callable] = None,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param record_info_names: The names of recording information
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    costs = []
    record_infos = {}
    for record_info_name in record_info_names:
        record_infos.update({record_info_name: []})
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            for i in range(len(_info)):
                if 'cost' in _info[i].keys():
                    costs.append(_info[i]['cost'])
                else:
                    costs = None
                for record_info_name in record_info_names:
                    if record_info_name == 'ego_velocity_x':
                        record_infos[record_info_name].append(np.mean(_info[i]['ego_velocity'][0]))
                    elif record_info_name == 'ego_velocity_y':
                        record_infos[record_info_name].append(np.mean(_info[i]['ego_velocity'][1]))
                    else:
                        record_infos[record_info_name].append(np.mean(_info[i][record_info_name]))
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, record_infos, costs
