import copy
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
from common.cns_config import get_cns_config
from models.constraint_net.mixture_constraint_net import MixtureConstraintNet
from cirl_stable_baselines3.ppo_lag.ma_ppo_lag import MultiAgentPPOLagrangian
from common.cns_visualization import constraint_visualization_2d, constraint_visualization_1d
from models.constraint_net.info_constraint_net import InfoConstraintNet
from common.cns_evaluation import evaluate_meicrl_policy
from common.cns_env import make_train_env, make_eval_env
from cirl_stable_baselines3.common import logger
from cirl_stable_baselines3.common.vec_env import sync_envs_normalization, VecNormalize
from utils.data_utils import read_args, load_config, ProgressBarManager, del_and_make, load_expert_data, \
    get_input_features_dim, process_memory, print_resource
from utils.env_utils import check_if_duplicate_seed
from utils.model_utils import load_ppo_config, build_code
from common.cns_sample import sample_from_multi_agents
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
    # debug_msg = ''
    debug_msg = 'robust_check_'  # 'robust_check_', 'sanity_check-', 'semi_check-'
    if 'robust_check' in debug_msg:
        debug_msg += str(config['running']['robust_weight']) + '_'
    # config['CN']['latent_dim'] = 1
    if debug_mode:
        # config['device'] = 'cpu'
        config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
        config['PPO']['forward_timesteps'] = 3000  # 2000
        config['PPO']['n_steps'] = 500
        config['PPO']['n_epochs'] = 2
        config['running']['n_eval_episodes'] = 2
        config['running']['save_every'] = 1
        config['running']['sample_rollouts'] = 2
        # config['running']['sample_data_num'] = 500
        # config['running']['store_sample_num'] = 1000
        # config['CN']['cn_batch_size'] = 3
        config['CN']['backward_iters'] = 2
        debug_msg += 'debug-'
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
                       latent_info_str=config['env']['latent_info_str'],
                       latent_dim=config['CN']['latent_dim'],
                       reward_gamma=config['env']['reward_gamma'],
                       cost_gamma=config['env']['cost_gamma'],
                       multi_env=multi_env,
                       part_data=partial_data,
                       constraint_id=config['env']['constraint_id'],
                       max_seq_len=config['running']['max_seq_length'],
                       log_file=log_file,
                       )
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
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
                      cost_info_str=config['env']['cost_info_str'],
                      latent_info_str=config['env']['latent_info_str'],
                      normalize_obs=not config['env']['dont_normalize_obs'],
                      latent_dim=config['CN']['latent_dim'],
                      part_data=partial_data,
                      multi_env=sample_multi_env,
                      constraint_id=config['env']['constraint_id'],
                      max_seq_len=config['running']['max_seq_length'],
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
                      latent_info_str=config['env']['latent_info_str'],
                      latent_dim=config['CN']['latent_dim'],
                      part_data=partial_data,
                      multi_env=False,
                      constraint_id=config['env']['constraint_id'],
                      max_seq_len=config['running']['max_seq_length'],
                      log_file=log_file)

    mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                         time_prev=time_prev,
                                         process_name='Loading environment',
                                         log_file=log_file)

    # Load expert data
    expert_path = config['running']['expert_path']
    if debug_mode:
        expert_path = expert_path.replace('expert_data/', 'expert_data/debug_')
    expert_rollouts = config['running']['expert_rollouts']
    (expert_obs, expert_acs, expert_rs, expert_codes), expert_mean_reward = load_expert_data(
        expert_path=expert_path,
        num_rollouts=expert_rollouts,
        store_by_game=config['running']['store_by_game'],
        add_latent_code=True,
        log_file=log_file
    )

    # plot expert traj
    plt.figure(figsize=(6, 6))
    for record_info_idx in range(len(config['env']["record_info_names"])):
        record_info_name = config['env']["record_info_names"][record_info_idx]
        if config['running']['store_by_game']:
            for c_id in range(config['CN']['latent_dim']):
                data_indices = np.where(np.concatenate(expert_codes, axis=0)[:, c_id] == 1)[0]
                # tmp = np.concatenate(expert_obs, axis=0)[:, record_info_idx][data_indices]
                plt.hist(np.concatenate(expert_obs, axis=0)[:, record_info_idx][data_indices],
                         bins=40,
                         # range=(config['env']["visualize_info_ranges"][record_info_idx][0],
                         #        config['env']["visualize_info_ranges"][record_info_idx][1]),
                         label=c_id)
        else:
            for c_id in range(config['CN']['latent_dim']):
                data_indices = np.where(expert_codes[:, c_id] == 1)[0]
                plt.hist(expert_obs[:, record_info_idx][data_indices],
                         bins=40,
                         # range=(config['env']["visualize_info_ranges"][record_info_idx][0],
                         #        config['env']["visualize_info_ranges"][record_info_idx][1])
                         )
        plt.legend()
        plt.savefig(os.path.join(save_model_mother_dir, "{0}_expert_traj_visual.png".format(record_info_name)))

    # Logger
    if log_file is None:
        icrl_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        icrl_logger = logger.HumanOutputFormat(log_file)

    # Init cns
    cn_parameters = get_cns_config(config, train_env, expert_obs, expert_acs, log_file)
    if 'InfoICRL' == config['group']:
        cn_parameters.update({'latent_dim': config['CN']['latent_dim']})
        constraint_net = InfoConstraintNet(**cn_parameters)
    elif 'MEICRL' == config['group']:
        cn_parameters.update({'latent_dim': config['CN']['latent_dim']})
        cn_parameters.update({'max_seq_length': config['running']['max_seq_length']})
        cn_parameters.update({'init_density': config['running']['init_density']})
        cn_parameters.update({'use_expert_negative': config['CN']['use_expert_negative']})
        cn_parameters.update({'sample_probing_points': config['CN']['sample_probing_points']})
        cn_parameters.update({'n_probings': config['CN']['n_probings']})
        cn_parameters.update({'reverse_probing': config['CN']['reverse_probing']})
        cn_parameters.update({'negative_weight': config['CN']['negative_weight']})
        cn_parameters.update({"log_cost": config['PPO']['log_cost']})
        constraint_net = MixtureConstraintNet(**cn_parameters)
    else:
        raise ValueError("Unknown group: {0}".format(config['group']))

    # Init ppo agent
    nominal_agents = {}
    create_nominal_agent_functions = {}
    for aid in range(config['CN']['latent_dim']):
        config['CN']['cid'] = aid
        if 'sanity_check-' in debug_msg:
            config['CN']['contrastive_weight'] = 0.0
        ppo_parameters = load_ppo_config(config, train_env, seed, log_file)
        create_nominal_agent = lambda: MultiAgentPPOLagrangian(**ppo_parameters)
        create_nominal_agent_functions.update({aid: create_nominal_agent})
        nominal_agent = create_nominal_agent()
        nominal_agents.update({aid: nominal_agent})

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
            for aid in range(config['CN']['latent_dim']):
                nominal_agents[aid] = create_nominal_agent_functions[aid]
        current_progress_remaining = 1 - float(itr) / float(config['running']['n_iters'])

        # Pass constraint net cost function to cost wrapper
        train_env.set_cost_function(constraint_net.cost_function_with_code)
        eval_env.set_cost_function(constraint_net.cost_function_with_code)
        sample_env.set_cost_function(constraint_net.cost_function_with_code)
        train_env.set_latent_function(constraint_net.latent_function)
        eval_env.set_latent_function(constraint_net.latent_function)
        sample_env.set_latent_function(constraint_net.latent_function)
        train_env.games_by_aids = constraint_net.games_by_aids
        eval_env.games_by_aids = constraint_net.games_by_aids
        sample_env.games_by_aids = constraint_net.games_by_aids

        # Update agent
        forward_metrics_all = {}
        with ProgressBarManager(config['PPO']['forward_timesteps']) as callback:
            # for aid in [0]:
            for aid in range(config['CN']['latent_dim']):
                nominal_agents[aid].learn(
                    total_timesteps=config['PPO']['forward_timesteps'],
                    cost_info_str=config['env']['cost_info_str'],
                    latent_info_str=config['env']['latent_info_str'],
                    callback=[callback],
                )
                forward_metrics_all.update({aid: copy.copy(logger.Logger.CURRENT.name_to_value)})
                timesteps += nominal_agents[aid].num_timesteps

        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Training PPO model',
                                             log_file=log_file)

        # Sample nominal trajectories
        sync_envs_normalization(train_env, sample_env)
        sample_parameters = {}

        sample_data = sample_from_multi_agents(
            agents=nominal_agents,
            latent_dim=config['CN']['latent_dim'],
            env=sample_env,
            deterministic=False,
            rollouts=int(config['running']['sample_rollouts']),
            store_by_game=config['running']['store_by_game'],
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

        # add the background information
        other_cn_parameters = {}
        other_cn_parameters['nominal_codes'] = sample_codes
        # TODO: the expert codes are for temporal usage. Remove it in the formal implementation.
        other_cn_parameters['expert_codes'] = expert_codes
        other_cn_parameters['debug_msg'] = debug_msg,
        backward_metrics = constraint_net.train_nn(iterations=config['CN']['backward_iters'],
                                                   nominal_obs=sample_obs,
                                                   nominal_acs=sample_acts,
                                                   episode_lengths=lengths,
                                                   obs_mean=mean,
                                                   obs_var=var,
                                                   current_progress_remaining=current_progress_remaining,
                                                   **other_cn_parameters)

        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Training CN model',
                                             log_file=log_file)

        # Evaluate:
        # reward on true environment
        sync_envs_normalization(train_env, eval_env)
        # model saving path
        save_path = save_model_mother_dir + '/model_{0}_itrs'.format(itr)
        if itr % config['running']['save_every'] == 0:
            del_and_make(save_path)
            # visualized sampled data
            plt.figure(figsize=(10, 10))
            for record_info_idx in range(len(config['env']["record_info_names"])):
                record_info_name = config['env']["record_info_names"][record_info_idx]
                if config['running']['store_by_game']:
                    for c_id in range(config['CN']['latent_dim']):
                        data_indices = np.where(np.concatenate(sample_codes, axis=0)[:, c_id] == 1)[0]
                        # tmp = np.concatenate(expert_obs, axis=0)[:, record_info_idx][data_indices]
                        plt.hist(np.concatenate(sample_obs, axis=0)[:, record_info_idx][data_indices],
                                 bins=40,
                                 # range=(config['env']["visualize_info_ranges"][record_info_idx][0],
                                 #        config['env']["visualize_info_ranges"][record_info_idx][1]),
                                 label=c_id)
                else:
                    for c_id in range(config['CN']['latent_dim']):
                        data_indices = np.where(sample_codes[:, c_id] == 1)[0]
                        plt.hist(sample_obs[:, record_info_idx][data_indices],
                                 bins=40,
                                 # range=(config['env']["visualize_info_ranges"][record_info_idx][0],
                                 #        config['env']["visualize_info_ranges"][record_info_idx][1])
                                 )
                plt.legend()
                plt.savefig(os.path.join(save_path, "{0}_sample_traj_visual.png".format(record_info_name)))

        mean_reward, std_reward, mean_nc_reward, std_nc_reward, record_infos, costs = \
            evaluate_meicrl_policy(models=nominal_agents,
                                   env=eval_env,
                                   latent_dim=config['CN']['latent_dim'],
                                   record_info_names=config['env']["record_info_names"],
                                   n_eval_episodes=config['running']['n_eval_episodes'],
                                   deterministic=True,
                                   # render=True if itr % config['running']['save_every'] == 0 else False,
                                   render=False,
                                   save_path=save_path,
                                   )
        mem_prev, time_prev = print_resource(mem_prev=mem_prev,
                                             time_prev=time_prev,
                                             process_name='Evaluation',
                                             log_file=log_file)

        # Save
        # (1) periodically
        if itr % config['running']['save_every'] == 0:
            for aid in range(config['CN']['latent_dim']):
                nominal_agents[aid].save(os.path.join(save_path, "nominal_agent_cid-{0}".format(aid)))
            constraint_net.save(os.path.join(save_path, "constraint_net"))
            if isinstance(train_env, VecNormalize):
                train_env.save(os.path.join(save_path, "train_env_stats.pkl"))

            if len(np.concatenate(expert_acs, axis=0).shape) == 1:
                empirical_input_means = np.concatenate([np.concatenate(expert_obs, axis=0),
                                                        np.expand_dims(np.concatenate(expert_acs, axis=0), 1)],
                                                       axis=1).mean(0)
            else:
                empirical_input_means = np.concatenate([np.concatenate(expert_obs, axis=0),
                                                        np.concatenate(expert_acs, axis=0)], axis=1).mean(0)
            for record_info_idx in range(len(config['env']["record_info_names"])):
                record_info_name = config['env']["record_info_names"][record_info_idx]
                for i in range(config['CN']['latent_dim']):
                    plot_record_infos, plot_costs = zip(*sorted(zip(record_infos[record_info_name][i], costs)))
                    constraint_visualization_1d(cost_function=constraint_net.cost_function_with_code,
                                                feature_range=config['env']["visualize_info_ranges"][record_info_idx],
                                                select_dim=config['env']["record_info_input_dims"][record_info_idx],
                                                obs_dim=constraint_net.obs_dim,
                                                acs_dim=1 if is_discrete else constraint_net.acs_dim,
                                                save_name=os.path.join(save_path,
                                                                       "cid-{0}_{1}_visual.png".
                                                                       format(i, record_info_name)),
                                                feature_data=plot_record_infos,
                                                feature_cost=plot_costs,
                                                feature_name=record_info_name,
                                                empirical_input_means=empirical_input_means,
                                                code_index=i,
                                                latent_dim=config['CN']['latent_dim'], )

        # (2) best
        if mean_nc_reward > best_true_reward:
            # print(utils.colorize("Saving new best model", color="green", bold=True), flush=True)
            print("Saving new best model", file=log_file, flush=True)
            for aid in range(config['CN']['latent_dim']):
                nominal_agents[aid].save(os.path.join(save_model_mother_dir, "best_nominal_agent_cid-{0}".format(aid)))
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
        for c_id in range(config['CN']['latent_dim']):
        # for c_id in [0, 1]:
            forward_metrics = forward_metrics_all[c_id]
            metrics.update(
                {"forward/" + 'aid:{0}/'.format(c_id) + k.replace("train/", ""): v for k, v in forward_metrics.items()})
        metrics.update(backward_metrics)
        print('\n\n', file=log_file, flush=True)

        # Log
        if config['verbose'] > 0:
            # icrl_logger.write(metrics, {k: None for k in metrics.keys()}, step=itr)
            icrl_logger.write(metrics, {k: str(metrics[k]) for k in metrics.keys()}, step=itr)


if __name__ == "__main__":
    args = read_args()
    train(args)
