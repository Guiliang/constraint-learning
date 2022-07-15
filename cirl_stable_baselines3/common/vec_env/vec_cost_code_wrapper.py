import pickle

from cirl_stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
import numpy as np
# =============================================================================
# Cost Wrapper
# =============================================================================
from utils.model_utils import update_code, build_code


class VecCostCodeWrapper(VecEnvWrapper):
    def __init__(self, venv, latent_dim, cost_info_str='cost'):
        super().__init__(venv)
        self.cost_info_str = cost_info_str
        self.latent_dim = latent_dim
        self.code_axis = [0 for i in range(self.num_envs)]

    def step_async(self, actions: np.ndarray):
        self.actions = actions
        self.venv.step_async(actions)

    def __getstate__(self):
        """
        Gets state for pickling.

        Excludes self.venv, as in general VecEnv's may not be pickleable."""
        state = self.__dict__.copy()
        # these attributes are not pickleable
        del state["venv"]
        del state["class_attributes"]
        # these attributes depend on the above and so we would prefer not to pickle
        del state["cost_function"]
        return state

    def __setstate__(self, state):
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None

    def set_venv(self, venv):
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        if self.venv is not None:
            raise ValueError("Trying to set venv of already initialized VecNormalize wrapper.")
        VecEnvWrapper.__init__(self, venv)

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        code = build_code(code_axis=self.code_axis, code_dim=self.latent_dim, num_envs=self.num_envs)
        for env_idx in range(self.num_envs):
            if news[env_idx]:
                # print('update code_axis')
                self.code_axis[env_idx] = update_code(code_dim=self.latent_dim, code_axis=self.code_axis[env_idx])
        if infos is None:
            infos = {}
        # Cost depends on previous observation and current actions
        cost = self.cost_function(self.previous_obs.copy(), self.actions.copy(), code.copy())  # [batch size]
        code_posterior = self.latent_function(self.previous_obs.copy(), self.actions.copy())  # [batch_size, code_dim]
        for i in range(len(infos)):
            infos[i][self.cost_info_str] = cost[i]
            infos[i]['code_posterior'] = code_posterior[i][self.code_axis[i]]
            infos[i]['code'] = code[i]
        self.previous_obs = obs.copy()
        return obs, rews, news, infos

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function

    def set_latent_function(self, latent_function):
        self.latent_function = latent_function

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.previous_obs = obs
        return obs

    @staticmethod
    def load(load_path: str, venv: VecEnv):
        """
        Loads a saved VecCostWrapper object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_cost_wrapper = pickle.load(file_handler)
        vec_cost_wrapper.set_venv(venv)
        return vec_cost_wrapper

    def save(self, save_path: str) -> None:
        """
        Save current VecCostWrapper object with
        all running statistics and settings (e.g. clip_obs)

        :param save_path: The path to save to
        """
        with open(save_path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    def get_image(self):
        print("abc")
        return None