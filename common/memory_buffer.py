import random
import numpy as np


class IRLDataQueue:
    def __init__(self, max_length=100000, seed=123):
        self.store_obs = []
        self.store_acts = []
        self.store_rs = []
        self.max_length = max_length
        random.seed(seed)

    def pop(self, pop_idx):
        del self.store_obs[pop_idx]
        del self.store_acts[pop_idx]
        del self.store_rs[pop_idx]

    def put(self, obs, acs, rs):
        for data_idx in range(len(obs)):
            if len(self.store_obs) >= self.max_length:
                rand_idx = random.randint(0, self.max_length - 1)
                self.pop(rand_idx)
            self.store_obs.append(obs[data_idx])
            self.store_acts.append(acs[data_idx])
            self.store_rs.append(rs[data_idx])

    def get(self, sample_num, store_by_game=False,):
        sample_obs = []
        sample_acs = []
        sample_rs = []
        data_len = 0
        while True:
            rand_idx = random.randint(0, len(self.store_obs) - 1)
            sample_obs.append(self.store_obs[rand_idx])
            sample_acs.append(self.store_acts[rand_idx])
            sample_rs.append(self.store_rs[rand_idx])
            if store_by_game:  # sample the trajectory of a game
                data_len += len(self.store_obs[rand_idx])
            else:  # sample a data point
                data_len += 1
            if data_len >= sample_num:
                break

        if store_by_game:
            return sample_obs, sample_acs, sample_rs
        else:
            return np.asarray(sample_obs), np.asarray(sample_acs), np.asarray(sample_rs)