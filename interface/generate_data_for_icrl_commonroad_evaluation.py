import argparse
import json
import logging
import os
import pickle
import time
from typing import Union, Callable
import numpy as np
import yaml
from gym import Env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from environment.commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv

from utils.data_utils import load_config, read_args, save_game_record

# def make_env(env_id, seed,  , info_keywords=()):
#     log_dir = 'icrl/test_log'
#
#     logging_path = 'icrl/test_log'
#
#     if log_dir is not None:
#         os.makedirs(log_dir, exist_ok=True)
#
#     def _init():
#         env = gym.make(env_id, logging_path=logging_path, **env_kwargs)
#         rank = 0
#         env.seed(seed + rank)
#         log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
#         env = Monitor(env, log_file, info_keywords=info_keywords)
#         return env
#
#     return _init
from utils.env_utils import make_env
from utils.plot_utils import pngs2gif


class CommonRoadVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        self.on_reset = None
        self.start_times = np.array([])

    def set_on_reset(self, on_reset_callback: Callable[[Env, float], None]):
        self.on_reset = on_reset_callback

    def reset(self):
        self.start_times = np.array([time.time()] * self.num_envs)
        return super().reset()

    def step_wait(self):
        out_of_scenarios = False
        for env_idx in range(self.num_envs):
            (obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx],) = self.envs[env_idx].step(
                self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs

                # Callback
                elapsed_time = time.time() - self.start_times[env_idx]
                self.on_reset(self.envs[env_idx], elapsed_time)
                self.start_times[env_idx] = time.time()

                # If one of the environments doesn't have anymore scenarios it will throw an Exception on reset()
                try:
                    obs = self.envs[env_idx].reset()
                except IndexError:
                    out_of_scenarios = True
            self._save_obs(env_idx, obs)
            self.buf_infos[env_idx]["out_of_scenarios"] = out_of_scenarios
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy()


LOGGER = logging.getLogger(__name__)


def create_environments(env_id: str, viz_path: str, test_path: str, model_path: str, num_threads: int = 1,
                        normalize=True, env_kwargs=None, testing_env=False, part_data=False) -> CommonRoadVecEnv:
    """
    Create CommonRoad vectorized environment
    """
    if viz_path is not None:
        env_kwargs.update({"visualization_path": viz_path})
    if testing_env:
        env_kwargs.update({"play": False})
        env_kwargs["test_env"] = True
    # else:
    #     env_kwargs.update({"play": True})
    # env_kwargs["test_env"] = True

    multi_env = True if num_threads > 1 else False
    if multi_env:
        env_kwargs['train_reset_config_path'] += '_split'
    if part_data:
        env_kwargs['train_reset_config_path'] += '_debug'
        env_kwargs['test_reset_config_path'] += '_debug'
        env_kwargs['meta_scenario_path'] += '_debug'
    # Create environment
    # note that CommonRoadVecEnv is inherited from DummyVecEnv
    # env = CommonRoadVecEnv([make_env(env_id, env_kwargs, rank=0, log_dir=test_path, seed=0)])
    envs = [make_env(env_id=env_id,
                     env_configs=env_kwargs,
                     rank=i,
                     log_dir=test_path,
                     multi_env=True if num_threads > 1 else False,
                     seed=0)
            for i in range(num_threads)]
    env = CommonRoadVecEnv(envs)

    def on_reset_callback(env: Union[Env, CommonroadEnv], elapsed_time: float):
        # reset callback called before resetting the env
        if env.observation_dict["is_goal_reached"][-1]:
            LOGGER.info("Goal reached")
        else:
            LOGGER.info("Goal not reached")
        env.render()

    env.set_on_reset(on_reset_callback)
    if normalize:
        LOGGER.info("Loading saved running average")
        vec_normalize_path = os.path.join(model_path, "train_env_stats.pkl")
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
        else:
            raise FileNotFoundError("vecnormalize.pkl not found in {0}".format(model_path))
        # env = VecNormalize(env, norm_obs=True, norm_reward=False)

    return env


def load_model(model_path: str):
    model_path = os.path.join(model_path, "best_nominal_model")
    model = PPO.load(model_path)
    return model


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug_mode", help="whether to use the debug mode",
                        dest="DEBUG_MODE",
                        default=False, required=False)
    parser.add_argument("-n", "--num_threads", help="number of threads for loading envs.",
                        dest="NUM_THREADS",
                        default=1, required=False)
    args = parser.parse_args()
    debug_mode = args.DEBUG_MODE
    num_threads = int(args.NUM_THREADS)

    # if log_file_path is not None:
    #     log_file = open(log_file_path, 'w')
    # else:
    log_file = None
    num_scenarios = 3000
    load_model_name = 'part-train_ppo_highD-Feb-01-2022-10:31'
    task_name = 'PPO-highD'
    data_generate_type = 'no-collision'
    if_testing_env = False
    if debug_mode:
        num_scenarios = 30

    model_loading_path = os.path.join('../save_model', task_name, load_model_name)
    with open(os.path.join(model_loading_path, 'model_hyperparameters.yaml')) as reader:
        config = yaml.safe_load(reader)

    print(json.dumps(config, indent=4), file=log_file, flush=True)

    # TODO: remove this line in the future
    if 'ppo' in config['env']['config_path']:
        config['env']['config_path'] = config['env']['config_path'].replace('_ppo', '')

    with open(config['env']['config_path'], "r") as config_file:
        env_configs = yaml.safe_load(config_file)

    evaluation_path = os.path.join('../evaluate_model', config['task'], load_model_name)
    if not os.path.exists(evaluation_path):
        os.mkdir(evaluation_path)
    # viz_path = os.path.join(evaluation_path, 'img')
    # if not os.path.exists(viz_path):
    #     os.mkdir(viz_path)

    save_expert_data_path = os.path.join('../data/expert_data/', '{0}{1}_{2}'.format(
        'debug_' if debug_mode else '',
        data_generate_type,
        load_model_name,
    ))
    if not os.path.exists(save_expert_data_path):
        os.mkdir(save_expert_data_path)

    env = create_environments(env_id="commonroad-v1",
                              viz_path=None,
                              test_path=evaluation_path,
                              model_path=model_loading_path,
                              num_threads=num_threads,
                              normalize=not config['env']['dont_normalize_obs'],
                              env_kwargs=env_configs,
                              testing_env=if_testing_env,
                              part_data=debug_mode)
    # TODO: this is for a quick check, maybe remove it in the future
    env.norm_reward = False

    model = load_model(model_loading_path)
    num_collisions, num_off_road, num_goal_reaching, num_timeout, total_scenarios = 0, 0, 0, 0, 0

    # In case there a no scenarios at all
    try:
        obs = env.reset()
    except IndexError:
        num_scenarios = 0

    success = 0
    benchmark_id_all = []
    while total_scenarios < num_scenarios:
        done, state = False, None
        benchmark_ids = [env.venv.envs[i].benchmark_id for i in range(num_threads)]
        save_data_flag = [True for i in range(num_threads)]
        for b_idx in range(len(benchmark_ids)):
            benchmark_id = benchmark_ids[b_idx]
            if benchmark_id in benchmark_id_all:
                print('skip game', benchmark_id, file=log_file, flush=True)
                save_data_flag[b_idx] = False
                # obs = env.reset()
                # continue
            else:
                benchmark_id_all.append(benchmark_id)
                print('senario id', benchmark_id, file=log_file, flush=True)

        # game_info_file = open(os.path.join(viz_path, benchmark_id, 'info_record.txt'), 'w')
        # game_info_file.write('current_step, is_collision, is_time_out, is_off_road, is_goal_reached\n')
        obs_all = [[] for i in range(num_threads)]
        original_obs_all = [[] for i in range(num_threads)]
        action_all = [[] for i in range(num_threads)]
        reward_all = [[] for i in range(num_threads)]
        reward_sums = [0 for i in range(num_threads)]
        running_steps = [0 for i in range(num_threads)]
        multi_thread_dones = [False for i in range(num_threads)]
        infos_done = []
        while not done:
            action, state = model.predict(obs, state=state, deterministic=False)
            new_obss, rewards, dones, infos = env.step(action)
            original_obs = env.get_original_obs() if isinstance(env, VecNormalize) else obs
            # benchmark_ids = [env.venv.envs[i].benchmark_id for i in range(num_threads)]
            # print(dones)
            # print(benchmark_ids)
            # save info
            for i in range(num_threads):
                if not multi_thread_dones[i]:
                    obs_all[i].append(obs[i])
                    original_obs_all[i].append(original_obs[i])
                    action_all[i].append(action[i])
                    reward_all[i].append(rewards[i])
                    running_steps[i] += 1
                    reward_sums[i] += rewards
                    if dones[i]:
                        infos_done.append(infos[i])
                        multi_thread_dones[i] = True
            # save_game_record(info[0], game_info_file)
            done = True
            for multi_thread_done in multi_thread_dones:
                if not multi_thread_done:
                    done = False
                    break
            obs = new_obss
        # game_info_file.close()

        # pngs2gif(png_dir=os.path.join(viz_path, benchmark_id))
        # log collision rate, off-road rate, and goal-reaching rate
        out_of_scenarios = True
        for i in range(num_threads):
            if not save_data_flag[i]:
                continue
            assert len(infos_done) == num_threads
            info = infos_done[i]
            total_scenarios += 1
            num_collisions += info["valid_collision"] if "valid_collision" in info else info["is_collision"]
            num_timeout += info.get("is_time_out", 0)
            num_off_road += info["valid_off_road"] if "valid_off_road" in info else info["is_off_road"]
            num_goal_reaching += info["is_goal_reached"]

            termination_reason = "other"
            if info.get("is_time_out", 0) == 1:
                termination_reason = "time_out"
            elif info.get("is_off_road", 0) == 1:
                termination_reason = "off_road"
            elif info.get("is_collision", 0) == 1:
                termination_reason = "collision"
            elif info.get("is_goal_reached", 0) == 1:
                termination_reason = "goal_reached"
            elif "is_over_speed" in info.keys() and info.get("is_over_speed", 0) == 1:
                termination_reason = "over_speed"

            if termination_reason not in data_generate_type:
                print('saving expert data for game {0} with terminal reason: {1}'.format(benchmark_ids[i],
                                                                                         termination_reason),
                      file=log_file, flush=True)

                saving_expert_data = {
                    'observations': np.asarray(obs_all[i]),
                    'actions': np.asarray(action_all[i]),
                    'original_observations': np.asarray(original_obs_all[i]),
                    'reward': np.asarray(reward_all[i]),
                    'reward_sum': reward_sums[i]
                }
                with open(os.path.join(save_expert_data_path,
                                       'scene-{0}_len-{1}.pkl'.format(benchmark_ids[i],
                                                                      running_steps[i])), 'wb') as file:
                    # A new file will be created
                    pickle.dump(saving_expert_data, file)

            if termination_reason == "goal_reached":
                print('{0}: goal reached'.format(benchmark_ids[i]), file=log_file, flush=True)
                success += 1
            if not info["out_of_scenarios"]:
                out_of_scenarios = False
        if out_of_scenarios:
            print('break because "out_of_scenarios"', file=log_file, flush=True)
            break
        else:
            obs = env.reset()

    print('total', total_scenarios, 'success', success, file=log_file, flush=True)


if __name__ == '__main__':
    # args = read_args()
    run()
