import datetime
import importlib
import json
import os
import sys
import time
import gym
import numpy as np
import yaml
from matplotlib import pyplot as plt

cwd = os.getcwd()
sys.path.append(cwd.replace('/interface', ''))
from cirl_stable_baselines3.ppo_lag.ppo_latent_lag import PPOLagrangianInfo
from common.cns_visualization import constraint_visualization_2d
from models.constraint_net.info_constraint_net import InfoConstraintNet
from common.cns_evaluation import evaluate_meicrl_policy
from common.cns_env import make_train_env, make_eval_env
from cirl_stable_baselines3.common import logger
from cirl_stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from utils.data_utils import read_args, load_config, ProgressBarManager, del_and_make, load_expert_data, \
    get_input_features_dim, process_memory, print_resource, load_expert_data_tmp
from utils.env_utils import multi_threads_sample_from_agent, sample_from_agent, get_obs_feature_names, is_mujoco, \
    check_if_duplicate_seed, is_commonroad
from utils.model_utils import get_net_arch, load_ppo_config, build_code
import warnings

warnings.filterwarnings("ignore")


def null_cost(x, *args):
    # Zero cost everywhere
    return np.zeros(x.shape[:1])


def train(config):
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
        config['device'] = 'cpu'
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['PPO']['forward_timesteps'] = 2000  # 2000
        config['PPO']['n_steps'] = 500
        config['PPO']['n_epochs'] = 2
        # config['running']['n_eval_episodes'] = 3
        config['running']['save_every'] = 1
        config['running']['sample_rollouts'] = 2
        # config['running']['sample_data_num'] = 500
        # config['running']['store_sample_num'] = 1000
        # config['CN']['cn_batch_size'] = 3
        config['CN']['backward_iters'] = 2
        debug_msg = 'debug-'
        partial_data = True
        # debug_msg += 'part-'
    if partial_data:
        debug_msg += 'part-'

    if num_threads is not None:
        config['env']['num_threads'] = int(num_threads)

    print(json.dumps(config, indent=4), file=log_file, flush=True)
    current_time_date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M')

    save_model_mother_dir = '{0}/{1}/{5}{2}{3}-{4}-seed_{6}/'.format(
        config['env']['save_dir'],
        config['task'],
        args.config_file.split('/')[-1].split('.')[0],
        '-multi_env' if multi_env else '',
        current_time_date,
        debug_msg,
        seed
    )

    skip_running = check_if_duplicate_seed(seed=seed,
                                           config=config,
                                           current_time_date=current_time_date,
                                           save_model_mother_dir=save_model_mother_dir,
                                           log_file=log_file)
    if skip_running:
        return

    if not os.path.exists('{0}/{1}/'.format(config['env']['save_dir'], config['task'])):
        os.mkdir('{0}/{1}/'.format(config['env']['save_dir'], config['task']))
    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)
    print("Saving to the file: {0}".format(save_model_mother_dir), file=log_file, flush=True)

    with open(os.path.join(save_model_mother_dir, "model_hyperparameters.yaml"), "w") as hyperparam_file:
        yaml.dump(config, hyperparam_file)

    # init computational resource
    mem_prev = process_memory()
    time_prev = start_time = time.time()

    # Create the vectorized environments
    train_env, env_configs = \
        make_train_env(env_id=config['env']['train_env_id'],
                       config_path=config['env']['config_path'],
                       save_dir=save_model_mother_dir,
                       group=config['group'],
                       base_seed=seed,
                       num_threads=num_threads,
                       use_cost=config['env']['use_cost'],
                       normalize_obs=not config['env']['dont_normalize_obs'],
                       normalize_reward=not config['env']['dont_normalize_reward'],
                       normalize_cost=not config['env']['dont_normalize_cost'],
                       cost_info_str=config['env']['cost_info_str'],
                       latent_dim=config['CN']['latent_dim'] if 'MEICRL' == config['group'] else None,
                       reward_gamma=config['env']['reward_gamma'],
                       cost_gamma=config['env']['cost_gamma'],
                       multi_env=multi_env,
                       part_data=partial_data,
                       log_file=log_file,
                       )
    all_obs_feature_names = get_obs_feature_names(train_env, config['env']['train_env_id'])
    print("The observed features are: {0}".format(all_obs_feature_names), file=log_file, flush=True)

    # We don't need cost when taking samples
    save_valid_mother_dir = os.path.join(save_model_mother_dir, "sample/")
    if not os.path.exists(save_valid_mother_dir):
        os.mkdir(save_valid_mother_dir)

    sample_num_threads = 1
    sample_multi_env = False
    sample_env, env_configs = \
        make_eval_env(env_id=config['env']['train_env_id'],
                      config_path=config['env']['config_path'],
                      save_dir=save_valid_mother_dir,
                      group=config['group'],
                      num_threads=sample_num_threads,
                      mode='sample',
                      use_cost=config['env']['use_cost'],
                      normalize_obs=not config['env']['dont_normalize_obs'],
                      latent_dim=config['CN']['latent_dim'] if 'MEICRL' == config['group'] else None,
                      part_data=partial_data,
                      multi_env=sample_multi_env,
                      log_file=log_file)

    save_test_mother_dir = os.path.join(save_model_mother_dir, "test/")
    if not os.path.exists(save_test_mother_dir):
        os.mkdir(save_test_mother_dir)
    eval_env, env_configs = \
        make_eval_env(env_id=config['env']['eval_env_id'],
                      config_path=config['env']['config_path'],
                      save_dir=save_test_mother_dir,
                      group=config['group'],
                      num_threads=1,
                      mode='test',
                      use_cost=config['env']['use_cost'],
                      normalize_obs=not config['env']['dont_normalize_obs'],
                      cost_info_str=config['env']['cost_info_str'],
                      latent_dim=config['CN']['latent_dim'] if 'MEICRL' == config['group'] else None,
                      part_data=partial_data,
                      multi_env=False,
                      log_file=log_file)

    mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                         time_prev=time_prev,
                                         process_name='Loading environment',
                                         log_file=log_file)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    print('is_discrete', is_discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]
    action_low, action_high = None, None
    if isinstance(sample_env.action_space, gym.spaces.Box):
        action_low, action_high = sample_env.action_space.low, sample_env.action_space.high

    # Load expert data
    expert_path = config['running']['expert_path']
    if debug_mode:
        expert_path = expert_path.replace('expert_data/', 'expert_data/debug_')
    expert_rollouts = config['running']['expert_rollouts']
    (expert_obs, expert_acs, expert_rs, expert_cs), expert_mean_reward = load_expert_data(
        expert_path=expert_path,
        num_rollouts=expert_rollouts,
        store_by_game=config['running']['store_by_game'],
        add_latent_code=True,
        log_file=log_file
    )
    if config['running']['store_by_game']:
        expert_obs_mean = np.mean(np.concatenate(expert_obs, axis=0), axis=0).tolist()
    else:
        expert_obs_mean = np.mean(expert_obs, axis=0).tolist()

    # plot expert traj
    plt.figure(figsize=(6, 6))
    plt.scatter(expert_obs[:, -2], expert_obs[:, -1], s=10)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.grid()
    plt.savefig(os.path.join(save_model_mother_dir, "expert_traj_visual.png"))

    expert_obs_mean = ['%.5f' % elem for elem in expert_obs_mean]
    if len(all_obs_feature_names) == len(expert_obs_mean):
        expert_obs_name_mean = dict(zip(all_obs_feature_names, expert_obs_mean))
    else:
        expert_obs_name_mean = expert_obs_mean
    print("The expert features means are: {0}".format(expert_obs_name_mean),
          file=log_file,
          flush=True)

    # Logger
    if log_file is None:
        icrl_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        icrl_logger = logger.HumanOutputFormat(log_file)

    # Initialize constraint net, true constraint net
    cn_lr_schedule = lambda x: (config['CN']['anneal_clr_by_factor'] ** (config['running']['n_iters'] * (1 - x))) \
                               * config['CN']['cn_learning_rate']

    cn_obs_select_name = config['CN']['cn_obs_select_name']
    print("Selecting obs features are : {0}".format(cn_obs_select_name if cn_obs_select_name is not None else 'all'),
          file=log_file, flush=True)
    cn_obs_select_dim = get_input_features_dim(feature_select_names=cn_obs_select_name,
                                               all_feature_names=all_obs_feature_names)
    cn_acs_select_name = config['CN']['cn_acs_select_name']
    print("Selecting acs features are : {0}".format(cn_acs_select_name if cn_acs_select_name is not None else 'all'),
          file=log_file, flush=True)
    cn_acs_select_dim = get_input_features_dim(feature_select_names=cn_acs_select_name,
                                               all_feature_names=['a_ego_0', 'a_ego_1'] if is_commonroad(
                                                   env_id=config['env']['train_env_id']) else None)

    cn_parameters = {
        'obs_dim': obs_dim,
        'acs_dim': acs_dim,
        'hidden_sizes': config['CN']['cn_layers'],
        'batch_size': config['CN']['cn_batch_size'],
        'lr_schedule': cn_lr_schedule,
        'expert_obs': expert_obs,  # select obs at a time step t
        'expert_acs': expert_acs,  # select acs at a time step t
        'is_discrete': is_discrete,
        'regularizer_coeff': config['CN']['cn_reg_coeff'],
        'obs_select_dim': cn_obs_select_dim,
        'acs_select_dim': cn_acs_select_dim,
        'clip_obs': config['CN']['clip_obs'],
        'initial_obs_mean': None if not config['CN']['cn_normalize'] else np.zeros(obs_dim),
        'initial_obs_var': None if not config['CN']['cn_normalize'] else np.ones(obs_dim),
        'action_low': action_low,
        'action_high': action_high,
        'target_kl_old_new': config['CN']['cn_target_kl_old_new'],
        'target_kl_new_old': config['CN']['cn_target_kl_new_old'],
        'train_gail_lambda': config['CN']['train_gail_lambda'],
        'eps': config['CN']['cn_eps'],
        'device': config['device'],
        'task': config['task'],
    }

    if 'MEICRL' == config['group']:
        cn_parameters.update({'latent_dim': config['CN']['latent_dim'], })
        constraint_net = InfoConstraintNet(**cn_parameters)
    else:
        raise ValueError("Unknown group: {0}".format(config['group']))

    # Pass constraint net cost function to cost wrapper (train env)
    train_env.set_cost_function(constraint_net.cost_function_with_code)
    sample_env.set_cost_function(constraint_net.cost_function_with_code)
    # Pass encoder posterior to cost wrapper (train env)
    train_env.set_latent_function(constraint_net.latent_function)
    sample_env.set_latent_function(constraint_net.latent_function)

    # Init ppo agent
    ppo_parameters = load_ppo_config(config, train_env, seed, log_file)
    create_nominal_agent = lambda: PPOLagrangianInfo(**ppo_parameters)
    nominal_agent = create_nominal_agent()

    mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                         time_prev=time_prev,
                                         process_name='Setting model',
                                         log_file=log_file)
    # Train
    timesteps = 0.
    print("\nBeginning training", file=log_file, flush=True)
    best_true_reward, best_true_cost, best_forward_kl, best_reverse_kl = -np.inf, np.inf, np.inf, np.inf
    for itr in range(config['running']['n_iters']):
        if config['PPO']['reset_policy'] and itr != 0:
            print("\nResetting agent", file=log_file, flush=True)
            nominal_agent = create_nominal_agent()
        current_progress_remaining = 1 - float(itr) / float(config['running']['n_iters'])
        # Update agent
        with ProgressBarManager(config['PPO']['forward_timesteps']) as callback:
            nominal_agent.learn(
                total_timesteps=config['PPO']['forward_timesteps'],
                cost_function="cost",  # Cost should come from cost wrapper
                callback=[callback]
            )
            forward_metrics = logger.Logger.CURRENT.name_to_value
            timesteps += nominal_agent.num_timesteps

        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Training PPO model',
                                             log_file=log_file)
        # Sample nominal trajectories
        sync_envs_normalization(train_env, sample_env)
        sample_parameters = {}

        sample_parameters['store_code'] = True
        if isinstance(sample_env, VecNormalize):
            init_codes = build_code(code_axis=sample_env.venv.code_axis,
                                    code_dim=sample_env.venv.latent_dim,
                                    num_envs=sample_env.venv.num_envs)
        else:
            init_codes = build_code(code_axis=sample_env.code_axis,
                                    code_dim=sample_env.latent_dim,
                                    num_envs=sample_env.num_envs)
        sample_parameters['init_codes'] = init_codes

        sample_data = sample_from_agent(
            agent=nominal_agent,
            env=sample_env,
            deterministic=True,
            rollouts=int(config['running']['sample_rollouts']),
            store_by_game=config['running']['use_buffer'],
            **sample_parameters
        )
        orig_observations, observations, actions, rewards, codes, sum_rewards, lengths = sample_data
        sample_obs = orig_observations
        sample_acts = actions
        sample_codes = codes
        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Sampling',
                                             log_file=log_file)
        # Update constraint net
        mean, var = None, None
        if config['CN']['cn_normalize']:
            mean, var = sample_env.obs_rms.mean, sample_env.obs_rms.var

        cns_parameters = {}
        cns_parameters['nominal_codes'] = sample_codes
        cns_parameters['expert_codes'] = expert_cs
        backward_metrics = constraint_net.train_nn(iterations=config['CN']['backward_iters'],
                                                   nominal_obs=sample_obs,
                                                   nominal_acs=sample_acts,
                                                   episode_lengths=lengths,
                                                   obs_mean=mean,
                                                   obs_var=var,
                                                   current_progress_remaining=current_progress_remaining,
                                                   **cns_parameters)

        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Training CN model',
                                             log_file=log_file)

        # Pass constraint net cost function to cost wrapper
        train_env.set_cost_function(constraint_net.cost_function_with_code)
        eval_env.set_cost_function(constraint_net.cost_function_with_code)
        sample_env.set_cost_function(constraint_net.cost_function_with_code)
        # Pass encoder posterior to cost wrapper (train env)
        train_env.set_latent_function(constraint_net.latent_function)
        eval_env.set_latent_function(constraint_net.latent_function)
        sample_env.set_latent_function(constraint_net.latent_function)

        # Evaluate:
        # reward on true environment
        sync_envs_normalization(train_env, eval_env)
        # model saving path
        save_path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
        if itr % config['running']['save_every'] == 0:
            del_and_make(save_path)
            # visualized sampled data
            plt.figure(figsize=(6, 6))
            plt.scatter(sample_obs[:, -2], sample_obs[:, -1], s=10)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.grid()
            plt.savefig(os.path.join(save_path, "sample_traj_visual.png"))
        mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs = \
            evaluate_meicrl_policy(nominal_agent, eval_env,
                                   record_info_names=[],
                                   n_eval_episodes=config['running']['n_eval_episodes'],
                                   deterministic=True,
                                   render=True if itr % config['running']['save_every'] == 0 else False,
                                   save_path=save_path,
                                   )
        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Evaluation',
                                             log_file=log_file)

        # Save
        # (1) periodically
        if itr % config['running']['save_every'] == 0:
            nominal_agent.save(os.path.join(save_path, "nominal_agent"))
            constraint_net.save(os.path.join(save_path, "constraint_net"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_path, "train_env_stats.pkl"))
            if len(expert_acs.shape) == 1:
                empirical_input_means = np.concatenate([expert_obs, np.expand_dims(expert_acs, 1)], axis=1).mean(0)
            else:
                empirical_input_means = np.concatenate([expert_obs, expert_acs], axis=1).mean(0)
            constraint_visualization_2d(cost_function_with_code=constraint_net.cost_function_with_code,
                                        feature_range=config['env']["visualize_info_ranges"],
                                        select_dims=config['env']["record_info_input_dims"],
                                        obs_dim=constraint_net.obs_dim,
                                        acs_dim=1 if is_discrete else constraint_net.acs_dim,
                                        latent_dim=constraint_net.latent_dim,
                                        empirical_input_means=empirical_input_means,
                                        save_path=save_path)

        # (2) best
        if mean_nc_reward > best_true_reward:
            # print(utils.colorize("Saving new best model", color="green", bold=True), flush=True)
            print("Saving new best model", file=log_file, flush=True)
            nominal_agent.save(os.path.join(save_model_mother_dir, "best_nominal_model"))
            constraint_net.save(os.path.join(save_model_mother_dir, "best_constraint_net_model"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_model_mother_dir, "train_env_stats.pkl"))

        # Update best metrics
        if mean_nc_reward > best_true_reward:
            best_true_reward = mean_nc_reward

        # Collect metrics
        metrics = {
            "running time(m)": (time.time() - start_time) / 60,
            # "env loading resource": load_env_str,
            # "model setting resource": set_model_str,
            # "train ppo resource": train_ppo_str,
            # "sample resource": sample_str,
            # "train cns resource": train_cn_str,
            # "evaluate resource": eval_str,
            "run_iter": itr,
            "timesteps": timesteps,
            "true/mean_nc_reward": mean_nc_reward,
            "true/std_nc_reward": std_nc_reward,
            "true/mean_reward": mean_reward,
            "true/std_reward": std_reward,
            "best_true/best_reward": best_true_reward
        }

        metrics.update({k.replace("train/", "forward/"): v for k, v in forward_metrics.items()})
        metrics.update(backward_metrics)

        # Log
        if config['verbose'] > 0:
            icrl_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)


if __name__ == "__main__":
    args = read_args()
    train(args)
