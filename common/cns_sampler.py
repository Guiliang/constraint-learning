import numpy as np

from planner.cross_entropy_method.cem import CEMAgent
from stable_baselines3.common import vec_env


class ConstrainedRLSampler:
    """
    Sampling based on the policy and planning
    """

    def __init__(self, rollouts, store_by_game, cost_info_str, env, planning_config=None):
        self.rollouts = rollouts
        self.store_by_game = store_by_game
        self.planning_config = planning_config
        self.env = env
        self.cost_info_str = cost_info_str
        self.policy_agent = None
        if self.planning_config is not None:
            self.apply_planning = True
            self.planner = CEMAgent(config=self.planning_config, env=self.env, cost_info_str=cost_info_str)
        else:
            self.apply_planning = False

    def sample_from_agent(self, policy_agent, new_env):
        if isinstance(new_env, vec_env.VecEnv):
            assert new_env.num_envs == 1, "You must pass only one environment when using this function"

        self.env = new_env
        self.policy_agent = policy_agent
        if self.apply_planning:
            self.planner.env = new_env

        if self.apply_planning:
            self.planner.plan(previous_actions=[], prior_policy=policy_agent)
        else:
            self.sample_with_policy()

    def sample_with_policy(self):
        all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
        sum_rewards, lengths = [], []
        for i in range(self.rollouts):
            # Avoid double reset, as VecEnv are reset automatically
            if i == 0:
                obs = self.env.reset()

            done, state = False, None
            episode_sum_reward = 0.0
            episode_length = 0

            origin_obs_game = []
            obs_game = []
            acs_game = []
            rs_game = []
            while not done:
                action, state = self.policy_agent.predict(obs, state=state, deterministic=False)
                origin_obs_game.append(self.env.get_original_obs())
                obs_game.append(obs)
                acs_game.append(action)
                if not self.store_by_game:
                    all_orig_obs.append(self.env.get_original_obs())
                    all_obs.append(obs)
                    all_acs.append(action)
                obs, reward, done, _info = self.env.step(action)
                rs_game.append(reward)
                if not self.store_by_game:
                    all_rs.append(reward)

                episode_sum_reward += reward
                episode_length += 1
            if self.store_by_game:
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

        if self.store_by_game:
            return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths
        else:
            all_orig_obs = np.squeeze(np.array(all_orig_obs), axis=1)
            all_obs = np.squeeze(np.array(all_obs), axis=1)
            all_acs = np.squeeze(np.array(all_acs), axis=1)
            all_rs = np.array(all_rs)
            sum_rewards = np.squeeze(np.array(sum_rewards), axis=1)
            lengths = np.array(lengths)
            return all_orig_obs, all_obs, all_acs, all_rs, sum_rewards, lengths