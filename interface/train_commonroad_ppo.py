import json
import os
import sys
import time
import gym
import numpy as np
import datetime
import yaml

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))

import environment.commonroad_rl.gym_commonroad  # this line must be included
from exploration.exploration import ExplorationRewardCallback
from stable_baselines3 import PPO
from stable_baselines3.common import logger
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize

from utils.data_utils import colorize, ProgressBarManager, del_and_make, read_args, load_config, process_memory
from utils.env_utils import make_train_env, make_eval_env
from utils.model_utils import get_net_arch
from config.config_commonroad import cfg

config = cfg()


def null_cost(x, *args):
    # Zero cost everywhere
    return np.zeros(x.shape[:1])


def train(args):
    config, debug_mode, log_file_path, partial_data, num_threads, seed = load_config(args)

    if num_threads > 1:
        multi_env = True
        config.update({'multi_env': True})
    else:
        multi_env = False
        config.update({'multi_env': False})

    if log_file_path is not None:
        log_file = open(log_file_path, 'w')
    else:
        log_file = None
    debug_msg = ''
    if debug_mode:
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['PPO']['forward_timesteps'] = 100  # 2000
        config['running']['n_eval_episodes'] = 10
        config['running']['save_every'] = 1
        debug_msg = 'debug-'
        partial_data = True
    if partial_data:
        debug_msg += 'part-'

    if num_threads is not None:
        config['env']['num_threads'] = num_threads

    print(json.dumps(config, indent=4), file=log_file, flush=True)
    current_time_date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M')
    # today = datetime.date.today()
    # currentTime = today.strftime("%b-%d-%Y-%h-%m")

    save_model_mother_dir = '{0}/{1}/{5}{2}{3}-{4}-seed_{6}/'.format(
        config['env']['save_dir'],
        config['task'],
        args.config_file.split('/')[-1].split('.')[0],
        '-multi_env' if multi_env else '',
        current_time_date,
        debug_msg,
        seed
    )

    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)

    with open(os.path.join(save_model_mother_dir, "model_hyperparameters.yaml"), "w") as hyperparam_file:
        yaml.dump(config, hyperparam_file)

    mem_prev = process_memory()
    # Create the vectorized environments
    train_env = make_train_env(env_id=config['env']['train_env_id'],
                               config_path=config['env']['config_path'],
                               save_dir=save_model_mother_dir,
                               base_seed=seed,
                               num_threads=config['env']['num_threads'],
                               use_cost=config['env']['use_cost'],
                               normalize_obs=not config['env']['dont_normalize_obs'],
                               normalize_reward=not config['env']['dont_normalize_reward'],
                               normalize_cost=not config['env']['dont_normalize_cost'],
                               cost_info_str=config['env']['cost_info_str'],
                               reward_gamma=config['env']['reward_gamma'],
                               cost_gamma=config['env']['cost_gamma'],
                               log_file=log_file,
                               part_data=partial_data,
                               multi_env=multi_env,
                               )

    save_test_mother_dir = os.path.join(save_model_mother_dir, "test/")
    if not os.path.exists(save_test_mother_dir):
        os.mkdir(save_test_mother_dir)

    eval_env = make_eval_env(env_id=config['env']['eval_env_id'],
                             config_path=config['env']['config_path'],
                             save_dir=save_test_mother_dir,
                             use_cost=config['env']['use_cost'],
                             normalize_obs=not config['env']['dont_normalize_obs'],
                             log_file=log_file,
                             part_data=partial_data)

    mem_loading_environment = process_memory()
    print("Loading environment consumed memory: {0}/{1}".format(float(mem_loading_environment - mem_prev) / 1000000,
                                                                float(mem_loading_environment) / 1000000
                                                                ),
          file=log_file, flush=True)
    mem_prev = mem_loading_environment

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    # print('is_discrete', is_discrete, file=log_file, flush=True)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    # Logger
    if log_file is None:
        ppo_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        ppo_logger = logger.HumanOutputFormat(log_file)

    create_ppo_agent = lambda: PPO(
        policy=config['PPO']['policy_name'],
        env=train_env,
        learning_rate=config['PPO']['learning_rate'],
        n_steps=config['PPO']['n_steps'],
        batch_size=config['PPO']['batch_size'],
        n_epochs=config['PPO']['n_epochs'],
        gamma=config['PPO']['reward_gamma'],
        gae_lambda=config['PPO']['reward_gae_lambda'],
        clip_range=config['PPO']['clip_range'],
        ent_coef=config['PPO']['ent_coef'],
        vf_coef=config['PPO']['reward_vf_coef'],
        max_grad_norm=config['PPO']['max_grad_norm'],
        use_sde=config['PPO']['use_sde'],
        sde_sample_freq=config['PPO']['sde_sample_freq'],
        target_kl=config['PPO']['target_kl'],
        verbose=config['verbose'],
        seed=seed,
        device=config['device'],
        policy_kwargs=dict(net_arch=get_net_arch(config)))

    ppo_agent = create_ppo_agent()

    # Callbacks
    all_callbacks = []
    if config['PPO']['use_curiosity_driven_exploration']:
        explorationCallback = ExplorationRewardCallback(obs_dim, acs_dim, device=config.device)
        all_callbacks.append(explorationCallback)

    # Warmup
    # Warmup
    timesteps = 0.
    if config['PPO']['warmup_timesteps']:
        # print(colorize("\nWarming up", color="green", bold=True), file=log_file, flush=True)
        print("\nWarming up", file=log_file, flush=True)
        with ProgressBarManager(config['PPO']['warmup_timesteps']) as callback:
            ppo_agent.learn(total_timesteps=config['PPO']['warmup_timesteps'],
                            callback=callback)
            timesteps += ppo_agent.num_timesteps

    mem_before_training = process_memory()
    print("Setting model consumed memory: {0}/{1}".format(float(mem_before_training - mem_prev) / 1000000,
                                                          float(mem_before_training) / 1000000
                                                          ),
          file=log_file, flush=True)
    mem_prev = mem_before_training

    # Train
    start_time = time.time()
    # print(colorize("\nBeginning training", color="green", bold=True), file=log_file, flush=True)
    print("\nBeginning training", file=log_file, flush=True)
    best_true_reward = -np.inf
    for itr in range(config['running']['n_iters']):
        if config['PPO']['reset_policy'] and itr != 0:
            # print(colorize("Resetting agent", color="green", bold=True), file=log_file, flush=True)
            print("Resetting agent", file=log_file, flush=True)
            ppo_agent = create_ppo_agent()

        current_progress_remaining = 1 - float(itr) / float(config['running']['n_iters'])

        # Update agent
        with ProgressBarManager(config['PPO']['forward_timesteps']) as callback:
            ppo_agent.learn(
                total_timesteps=config['PPO']['forward_timesteps'],
                callback=[callback] + all_callbacks
            )
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += ppo_agent.num_timesteps

        # Evaluate:
        # reward on true environment
        sync_envs_normalization(train_env, eval_env)
        average_true_reward, std_true_reward = evaluate_policy(ppo_agent, eval_env,
                                                               n_eval_episodes=config['running']['n_eval_episodes'],
                                                               deterministic=False)

        # Save
        # (1) periodically
        if itr % config['running']['save_every'] == 0:
            path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
            del_and_make(path)
            ppo_agent.save(os.path.join(path, "nominal_agent"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(path, "train_env_stats.pkl"))

        # (2) best
        if average_true_reward > best_true_reward:
            # print(colorize("Saving new best model", color="green", bold=True), flush=True, file=log_file)
            print("Saving new best model", flush=True, file=log_file)
            ppo_agent.save(os.path.join(save_model_mother_dir, "best_nominal_model"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_model_mother_dir, "train_env_stats.pkl"))

        # Update best metrics
        if average_true_reward > best_true_reward:
            best_true_reward = average_true_reward

        # Collect metrics
        metrics = {
            "time(m)": (time.time() - start_time) / 60,
            "run_iter": itr,
            "timesteps": timesteps,
            "true/reward": average_true_reward,
            "true/reward_std": std_true_reward,
            "best_true/best_reward": best_true_reward
        }

        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})

        # Log
        if config['verbose'] > 0:
            ppo_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)

        mem_during_training = process_memory()
        print("Training consumed memory: {0}/{1}".format(float(mem_during_training - mem_prev) / 1000000,
                                                         float(mem_during_training) / 1000000
                                                         ), file=log_file, flush=True)
        mem_prev = mem_during_training


if __name__ == "__main__":
    args = read_args()
    train(args)
