import math

import numpy as np
import torch
from torch.distributions import Normal
from planner.planning_agent import AbstractAgent, safe_deepcopy_env


class CEMAgent(AbstractAgent):
    """
        Cross-Entropy Method planner.
        The environment is copied and used as an oracle model to sample trajectories.
    """

    def __init__(self, env, config, cost_info_str='cost', store_by_game=False, eps=0.00001):
        super(CEMAgent, self).__init__(config)
        self.eps = eps
        self.env = env
        self.action_size = env.action_space.shape[0]
        self.cost_info_str = cost_info_str
        self.store_by_game = store_by_game

    @classmethod
    def default_config(cls):
        return dict(gamma=1.0,
                    horizon=10,
                    iterations=10,
                    candidates=100,
                    top_candidates=10,
                    std=0.1,
                    prior_lambda=1,
                    done_penalty=-1)

    def plan(self, previous_actions, prior_policy):
        action_distribution = Normal(
            loc=torch.zeros(self.config["horizon"], self.action_size),
            scale=torch.tensor(self.config["std"]).repeat(self.config["horizon"], self.action_size))
        all_orig_obs, all_obs, all_acs, all_rs = [], [], [], []
        sum_rewards, lengths = [], []
        best_orig_obs, best_obs, best_acs, best_rs, best_length, best_sum_rewards = None, None, None, None, None, None
        for i in range(self.config["iterations"]):
            sampled_actions = action_distribution.sample([self.config["candidates"]])
            returns = torch.zeros(self.config["candidates"])
            for c in range(self.config["candidates"]):
                obs = self.env.reset()
                state = None
                done = False
                origin_obs_game, obs_game, acs_game, rs_game = [], [], [], []
                for previous_action in previous_actions:
                    # pred_action, state = prior_policy.predict(obs, state=state, deterministic=False)
                    pred_action, state = prior_policy.predict(obs, state=state, deterministic=True)
                    obs, reward, done, _info = self.env.step(previous_action)
                for t in range(self.config["horizon"]):
                    if done or t == self.config["horizon"]-1:
                        lengths.append(t)
                        # returns[c] += self.config["done_penalty"]
                    else:
                        # prior_action, state = prior_policy.predict(obs, state=state, deterministic=False)
                        prior_action, state = prior_policy.predict(obs, state=state, deterministic=True)
                        sampled_action = sampled_actions[c, t, :]
                        prior_lambda = self.config['prior_lambda']
                        action = (sampled_action + prior_lambda * prior_action) / (1 + prior_lambda)
                        sampled_actions[c, t, :] = action
                        action = action.detach().numpy()
                        if i == self.config["iterations"] - 1:
                            origin_obs_game.append(self.env.get_original_obs())
                            obs_game.append(obs)
                            acs_game.append(action)
                        obs, reward, done, info = self.env.step(action)
                        if i == self.config["iterations"] - 1:
                            rs_game.append(reward)
                        returns[c] += self.config["gamma"] ** t * (reward +
                                                                   math.log(1 - info[0][self.cost_info_str]+self.eps))
                if i == self.config["iterations"] - 1:
                    all_orig_obs.append(np.squeeze(np.array(origin_obs_game), axis=1))
                    all_obs.append(np.squeeze(np.array(obs_game), axis=1))
                    # tmp = np.array(acs_game)
                    all_acs.append(np.squeeze(np.array(acs_game), axis=1))
                    all_rs.append(np.squeeze(np.asarray(rs_game)))
            # Re-fit belief to the K best action sequences
            _, topk = returns.topk(self.config["top_candidates"], largest=True, sorted=False)  # K ← argsort({R(j)}
            best_actions = sampled_actions[topk]
            if i == self.config["iterations"] - 1:
                best_orig_obs = [all_orig_obs[idx] for idx in topk]
                best_obs = [all_obs[idx] for idx in topk]
                best_acs = [all_acs[idx] for idx in topk]
                best_rs = [all_rs[idx] for idx in topk]
                best_length = [lengths[idx] for idx in topk]
                best_sum_rewards = [returns[idx] for idx in topk]
            # Update belief with new means and standard deviations
            mean = best_actions.mean(dim=0)
            std = best_actions.std(dim=0, unbiased=False)
            std = std.clip(min=1e-10, max=None)
            try:
                action_distribution = Normal(loc=mean, scale=std)
            except:
                for actions in best_actions:
                    for action in actions:
                        print(action)
                for std_point in std:
                    print(std_point)
                action_distribution = Normal(loc=mean, scale=std)
        # Return first action mean µ_t
        if self.store_by_game:
            return best_orig_obs, best_obs, best_acs, best_rs, best_length, best_sum_rewards
        else:
            best_orig_obs = np.squeeze(np.array(best_orig_obs), axis=1)
            best_obs = np.squeeze(np.array(best_obs), axis=1)
            best_acs = np.squeeze(np.array(best_acs), axis=1)
            best_rs = np.array(best_rs)
            best_sum_rewards = np.squeeze(np.array(best_sum_rewards), axis=1)
            best_length = np.array(best_length)
            return best_orig_obs, best_obs, best_acs, best_rs, best_length, best_sum_rewards

    def record(self, state, action, reward, next_state, done, info):
        pass

    def act(self, state):
        return self.plan(state, [])[0]

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def save(self, filename):
        return False

    def load(self, filename):
        return False


class PytorchCEMAgent(CEMAgent):
    """
    CEM planner with Recurrent state-space models (RSSM) for transition and rewards, as in PlaNet.
    Original implementation by Kai Arulkumaran from https://github.com/Kaixhin/PlaNet/blob/master/planner.py
    Allows batch forward of many candidates (e.g. 1000)
    """

    def __init__(self, env, config, transition_model, reward_model):
        super(CEMAgent, self).__init__(config)
        self.env = env
        self.action_size = env.action_space.shape[0]
        self.transition_model = transition_model
        self.reward_model = reward_model

    def plan(self, state, belief):
        belief, state = belief.expand(self.config["candidates"], -1), state.expand(self.config["candidates"], -1)
        # Initialize factorized belief over action sequences q(a_t:t+H) ← N(0, I)
        action_distribution = Normal(torch.zeros(self.config["horizon"], self.action_size, device=belief.device),
                                     torch.ones(self.config["horizon"], self.action_size, device=belief.device))
        for i in range(self.config["iterations"]):
            # Evaluate J action sequences from the current belief (in batch)
            beliefs, states = [belief], [state]
            actions = action_distribution.sample([self.config["candidates"]])  # Sample actions
            # Sample next states
            for t in range(self.config["horizon"]):
                next_belief, next_state, _, _ = self.transition_model(states[-1], actions[:, t], beliefs[-1])
                beliefs.append(next_belief)
                states.append(next_state)
            # Calculate expected returns (batched over time x batch)
            beliefs = torch.stack(beliefs[1:], dim=0).view(self.config["horizon"] * self.config["candidates"], -1)
            states = torch.stack(states[1:], dim=0).view(self.config["horizon"] * self.config["candidates"], -1)
            returns = self.reward_model(beliefs, states).view(self.config["horizon"], self.config["candidates"]).sum(
                dim=0)
            # Re-fit belief to the K best action sequences
            _, topk = returns.topk(self.config["top_candidates"], largest=True, sorted=False)  # K ← argsort({R(j)}
            best_actions = actions[topk]
            # Update belief with new means and standard deviations
            action_distribution = Normal(best_actions.mean(dim=0), best_actions.std(dim=0, unbiased=False))
        # Return first action mean µ_t
        return action_distribution.mean[0].to_list()
