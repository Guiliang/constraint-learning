import os

import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv

# =========================================================================== #
#                         Inverted Pendulum Wall                              #
# =========================================================================== #


class InvertedPendulumWall(InvertedPendulumEnv):

    def step(self, a):
        reward = 0.01
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        obs = self._get_obs()
        notdone = np.isfinite(obs).all() and (np.abs(obs[1]) <= 0.2)
        done = not notdone
        info = {'x_position': xposafter}
        return obs, reward, done, info
