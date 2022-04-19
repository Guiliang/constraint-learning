import datetime
import json
import os
import sys
import time

import yaml

from common.cns_env import make_train_env, make_eval_env
from stable_baselines3.common import logger
from utils.data_utils import read_args, load_config, process_memory, load_expert_data


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
        # config['device'] = 'cpu'
        # config['verbose'] = 2  # the verbosity level: 0 no output, 1 info, 2 debug
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

    if not os.path.exists('{0}/{1}/'.format(config['env']['save_dir'], config['task'])):
        os.mkdir('{0}/{1}/'.format(config['env']['save_dir'], config['task']))
    if not os.path.exists(save_model_mother_dir):
        os.mkdir(save_model_mother_dir)

    with open(os.path.join(save_model_mother_dir, "model_hyperparameters.yaml"), "w") as hyperparam_file:
        yaml.dump(config, hyperparam_file)

    mem_prev = process_memory()
    time_prev = time.time()
    # Create the vectorized environments
    train_env = make_train_env(env_id=config.train_env_id,
                                     save_dir=config.save_dir,
                                     use_cost_wrapper=False,
                                     base_seed=config.seed,
                                     num_threads=config.num_threads,
                                     normalize_obs=not config.dont_normalize_obs,
                                     normalize_reward=not config.dont_normalize_reward,
                                     normalize_cost=False,
                                     reward_gamma=config.reward_gamma
                                     )


    eval_env = make_eval_env(env_id=config.eval_env_id,
                                   use_cost_wrapper=False,
                                   normalize_obs=not config.dont_normalize_obs)

    # Set specs
    is_discrete = isinstance(train_env.action_space, gym.spaces.Discrete)
    obs_dim = train_env.observation_space.shape[0]
    acs_dim = train_env.action_space.n if is_discrete else train_env.action_space.shape[0]

    action_low, action_high = None, None
    if isinstance(train_env.action_space, gym.spaces.Box):
        action_low, action_high = train_env.action_space.low, train_env.action_space.high

    # Load expert data
    (expert_obs, expert_acs), expert_mean_reward = load_expert_data(expert_path=config.expert_path,
                                                                    num_rollouts=config.expert_rollouts)
    # expert_agent = PPOLagrangian.load(os.path.join(config.expert_path, "files/best_model.zip"))

    # Logger
    # Logger
    if log_file is None:
        gail_logger = logger.HumanOutputFormat(sys.stdout)
    else:
        gail_logger = logger.HumanOutputFormat(log_file)

    # Do we want to restore gail from a saved model?
    if config.gail_path is not None:
        discriminator = GailDiscriminator.load(
            config.gail_path,
            obs_dim=obs_dim,
            acs_dim=acs_dim,
            is_discrete=is_discrete,
            expert_obs=expert_obs,
            expert_acs=expert_acs,
            obs_select_dim=config.disc_obs_select_dim,
            acs_select_dim=config.disc_acs_select_dim,
            clip_obs=None,
            obs_mean=None,
            obs_var=None,
            action_low=action_low,
            action_high=action_high,
            device=config.device,
        )
        discriminator.freeze_weights = config.freeze_gail_weights
    else:  # Initialize GAIL and setup its callback
        discriminator = GailDiscriminator(
            obs_dim,
            acs_dim,
            config.disc_layers,
            config.disc_batch_size,
            get_schedule_fn(config.disc_learning_rate),
            expert_obs,
            expert_acs,
            is_discrete,
            config.disc_obs_select_dim,
            config.disc_acs_select_dim,
            clip_obs=config.clip_obs,
            initial_obs_mean=None,
            initial_obs_var=None,
            action_low=action_low,
            action_high=action_high,
            num_spurious_features=config.num_spurious_features,
            freeze_weights=config.freeze_gail_weights,
            eps=config.disc_eps,
            device=config.device
        )

    true_cost_function = get_true_cost_function(config.eval_env_id)

    if config.use_cost_shaping_callback:
        costShapingCallback = CostShapingCallback(true_cost_function,
                                                  obs_dim,
                                                  acs_dim,
                                                  use_nn_for_shaping=config.use_cost_net)
        all_callbacks = [costShapingCallback]
    else:
        plot_disc = True if config.train_env_id in ['DD2B-v0', 'DD3B-v0', 'CDD2B-v0', 'CDD3B-v0'] else False
        if config.disc_obs_select_dim is not None and config.disc_acs_select_dim is not None:
            plot_disc = True if (len(config.disc_obs_select_dim) < 3
                                 and config.disc_acs_select_dim[0] == -1) else False
        gail_update = GailCallback(discriminator, config.learn_cost, true_cost_function,
                                   config.save_dir, plot_disc=plot_disc)
        all_callbacks = [gail_update]

    # Define and train model
    model = PPO(
        policy=config.policy_name,
        env=train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.reward_gamma,
        gae_lambda=config.reward_gae_lambda,
        clip_range=config.clip_range,
        clip_range_vf=config.clip_range_reward_vf,
        ent_coef=config.ent_coef,
        vf_coef=config.reward_vf_coef,
        max_grad_norm=config.max_grad_norm,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        target_kl=config.target_kl,
        seed=config.seed,
        device=config.device,
        verbose=config.verbose,
        policy_kwargs=dict(net_arch=utils.get_net_arch(config))
    )

    # All callbacks
    save_periodically = callbacks.CheckpointCallback(
        config.save_every, os.path.join(config.save_dir, "models"),
        verbose=0
    )
    save_env_stats = utils.SaveEnvStatsCallback(train_env, config.save_dir)
    save_best = callbacks.EvalCallback(
        eval_env, eval_freq=config.eval_every, deterministic=False,
        best_model_save_path=config.save_dir, verbose=0,
        callback_on_new_best=save_env_stats
    )
    plot_func = get_plot_func(config.train_env_id)
    plot_callback = utils.PlotCallback(
        plot_func, train_env_id=config.train_env_id,
        plot_freq=config.plot_every, plot_save_dir=config.save_dir
    )

    # Organize all callbacks in list
    all_callbacks.extend([save_periodically, save_best, plot_callback])

    # Train
    model.learn(total_timesteps=int(config.timesteps),
                callback=all_callbacks)

    # Save final discriminator
    if not config.freeze_gail_weights:
        discriminator.save(os.path.join(config.save_dir, "gail_discriminator.pt"))

    # Save normalization stats
    if isinstance(train_env, VecNormalize):
        train_env.save(os.path.join(config.save_dir, "train_env_stats.pkl"))

    # Make video of final model
    if not config.wandb_sweep:
        sync_envs_normalization(train_env, eval_env)
        utils.eval_and_make_video(eval_env, model, config.save_dir, "final_policy")

    if config.sync_wandb:
        utils.sync_wandb(config.save_dir, 120)


if __name__ == "__main__":
    args = read_args()
    train(args)
