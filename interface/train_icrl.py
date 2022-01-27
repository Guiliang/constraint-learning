import importlib
import json
import os
import pickle
import sys
import time
import random

import gym
import numpy as np
from stable_baselines3 import PPOLagrangian
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from tqdm import tqdm

import commonroad_rl.gym_commonroad

import icrl.utils as utils
from icrl.plot_utils import plot_obs_ant
from icrl.exploration import ExplorationRewardCallback, LambdaShapingCallback
from icrl.constraint_net import ConstraintNet, plot_constraints

from icrl.config_commonroad import cfg

config = cfg()


def null_cost(x, *args):
    # Zero cost everywhere
    return np.zeros(x.shape[:1])


def load_expert_data(expert_path, num_rollouts):
    file_names = [i for i in range(29)]
    sample_names = random.sample(file_names, num_rollouts)

    expert_mean_reward = []
    for i in range(num_rollouts):
        file_name = sample_names[i]

        with open(os.path.join(expert_path, "%s.pkl" % str(file_name)), "rb") as f:
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


def icrl(config):
    # We only want to use cost wrapper for custom environments
    use_cost_wrapper_train = True
    use_cost_wrapper_eval = False

    # Create the vectorized environments
    train_env = utils.make_train_env(env_id=config.train_env_id,
                                     save_dir=config.save_dir,
                                     use_cost_wrapper=use_cost_wrapper_train,
                                     base_seed=config.seed,
                                     num_threads=config.num_threads,
                                     normalize_obs=not config.dont_normalize_obs,
                                     normalize_reward=not config.dont_normalize_reward,
                                     normalize_cost=not config.dont_normalize_cost,
                                     cost_info_str=config.cost_info_str,
                                     reward_gamma=config.reward_gamma,
                                     cost_gamma=config.cost_gamma)

    # We don't need cost when taking samples
    sampling_env = utils.make_eval_env(env_id=config.train_env_id,
                                       use_cost_wrapper=False,
                                       normalize_obs=not config.dont_normalize_obs)

    eval_env = utils.make_eval_env(env_id=config.eval_env_id,
                                   use_cost_wrapper=use_cost_wrapper_eval,
                                   normalize_obs=not config.dont_normalize_obs)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    print('is_discrete', is_discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(sampling_env.action_space, gym.spaces.Box):
        action_low, action_high = sampling_env.action_space.low, sampling_env.action_space.high

    # Load expert data
    (expert_obs, expert_acs), expert_mean_reward = load_expert_data(config.expert_path, config.expert_rollouts)

    # Logger
    icrl_logger = logger.HumanOutputFormat(sys.stdout)

    # Initialize constraint net, true constraint net
    cn_lr_schedule = lambda x: (config.anneal_clr_by_factor ** (config.n_iters * (1 - x))) * config.cn_learning_rate
    constraint_net = ConstraintNet(
        obs_dim,
        acs_dim,
        config.cn_layers,
        config.cn_batch_size,
        cn_lr_schedule,
        expert_obs,
        expert_acs,
        is_discrete,
        config.cn_reg_coeff,
        config.cn_obs_select_dim,
        config.cn_acs_select_dim,
        no_importance_sampling=config.no_importance_sampling,
        per_step_importance_sampling=config.per_step_importance_sampling,
        clip_obs=config.clip_obs,
        initial_obs_mean=None if not config.cn_normalize else np.zeros(obs_dim),
        initial_obs_var=None if not config.cn_normalize else np.ones(obs_dim),
        action_low=action_low,
        action_high=action_high,
        target_kl_old_new=config.cn_target_kl_old_new,
        target_kl_new_old=config.cn_target_kl_new_old,
        train_gail_lambda=config.train_gail_lambda,
        eps=config.cn_eps,
        device=config.device
    )

    # Pass constraint net cost function to cost wrapper (train env)
    train_env.set_cost_function(constraint_net.cost_function)

    # Initialize agent
    create_nominal_agent = lambda: PPOLagrangian(
        policy=config.policy_name,
        env=train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        reward_gamma=config.reward_gamma,
        reward_gae_lambda=config.reward_gae_lambda,
        cost_gamma=config.cost_gamma,
        cost_gae_lambda=config.cost_gae_lambda,
        clip_range=config.clip_range,
        clip_range_reward_vf=config.clip_range_reward_vf,
        clip_range_cost_vf=config.clip_range_cost_vf,
        ent_coef=config.ent_coef,
        reward_vf_coef=config.reward_vf_coef,
        cost_vf_coef=config.cost_vf_coef,
        max_grad_norm=config.max_grad_norm,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        target_kl=config.target_kl,
        penalty_initial_value=config.penalty_initial_value,
        penalty_learning_rate=config.penalty_learning_rate,
        budget=config.budget,
        seed=config.seed,
        device=config.device,
        verbose=0,
        pid_kwargs=dict(alpha=config.budget,
                        penalty_init=config.penalty_initial_value,
                        Kp=config.proportional_control_coeff,
                        Ki=config.integral_control_coeff,
                        Kd=config.derivative_control_coeff,
                        pid_delay=config.pid_delay,
                        delta_p_ema_alpha=config.proportional_cost_ema_alpha,
                        delta_d_ema_alpha=config.derivative_cost_ema_alpha, ),
        policy_kwargs=dict(net_arch=utils.get_net_arch(config))
    )

    nominal_agent = create_nominal_agent()

    # Callbacks
    all_callbacks = []
    if config.use_curiosity_driven_exploration:
        explorationCallback = ExplorationRewardCallback(obs_dim, acs_dim, device=config.device)
        all_callbacks.append(explorationCallback)

    # Warmup
    timesteps = 0.
    if config['PPO']['warmup_timesteps']:
        print(utils.colorize("\nWarming up", color="green", bold=True))
        with utils.ProgressBarManager(config['PPO']['warmup_timesteps']) as callback:
            nominal_agent.learn(total_timesteps=config['PPO']['warmup_timesteps'],
                                cost_function=null_cost,  # During warmup we dont want to incur any cost
                                callback=callback)
            timesteps += nominal_agent.num_timesteps

    # Train
    start_time = time.time()
    print(utils.colorize("\nBeginning training", color="green", bold=True), flush=True)
    best_true_reward, best_true_cost, best_forward_kl, best_reverse_kl = -np.inf, np.inf, np.inf, np.inf
    for itr in range(config['PPO']['n_iters']):
        if config['PPO']['reset_policy'] and itr != 0:
            print(utils.colorize("Resetting agent", color="green", bold=True), flush=True)
            nominal_agent = create_nominal_agent()
        current_progress_remaining = 1 - float(itr) / float(config['PPO']['n_iters'])

        # Update agent
        with utils.ProgressBarManager(config['PPO']['forward_timesteps']) as callback:
            nominal_agent.learn(
                total_timesteps=config['PPO']['forward_timesteps'],
                cost_function="",  # Cost should come from cost wrapper
                callback=[callback] + all_callbacks
            )
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += nominal_agent.num_timesteps

        # Sample nominal trajectories
        sync_envs_normalization(train_env, sampling_env)
        orig_observations, observations, actions, rewards, lengths = utils.sample_from_agent(
            nominal_agent, sampling_env, config.expert_rollouts)

        # Update constraint net
        mean, var = None, None
        if config.cn_normalize:
            mean, var = sampling_env.obs_rms.mean, sampling_env.obs_rms.var
        backward_metrics = constraint_net.train(config.backward_iters, orig_observations, actions, lengths,
                                                mean, var, current_progress_remaining)

        # Pass updated cost_function to cost wrapper (train_env)
        train_env.set_cost_function(constraint_net.cost_function)

        # Evaluate:
        # reward on true environment
        sync_envs_normalization(train_env, eval_env)
        average_true_reward, std_true_reward = evaluate_policy(nominal_agent, eval_env, n_eval_episodes=10,
                                                               deterministic=False)

        # Save
        # (1) periodically
        if itr % config.save_every == 0:
            path = os.path.join(config.save_dir, f"models/icrl_{itr}_itrs")
            utils.del_and_make(path)
            nominal_agent.save(os.path.join(path, f"nominal_agent"))
            constraint_net.save(os.path.join(path, f"cn.pt"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(path, f"{itr}_train_env_stats.pkl"))

        # (2) best
        if average_true_reward > best_true_reward:
            print(utils.colorize("Saving new best model", color="green", bold=True), flush=True)
            nominal_agent.save(os.path.join(config.save_dir, "best_nominal_model"))
            constraint_net.save(os.path.join(config.save_dir, "best_cn_model.pt"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(config.save_dir, "train_env_stats.pkl"))

        # Update best metrics
        if average_true_reward > best_true_reward:
            best_true_reward = average_true_reward

        # Collect metrics
        metrics = {
            "time(m)": (time.time() - start_time) / 60,
            "iteration": itr,
            "timesteps": timesteps,
            "true/reward": average_true_reward,
            "true/reward_std": std_true_reward,
            "best_true/best_reward": best_true_reward
        }

        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})
        metrics.update(backward_metrics)

        # Log
        if config.verbose > 0:
            icrl_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)


icrl(config)
