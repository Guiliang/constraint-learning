from functools import partial
import numpy as np


def get_true_cost_function(env_id, env_configs={}):
    """Returns the cost function correpsonding to provided env)"""
    if 'constraint_id' in env_configs:
        c_id = env_configs['constraint_id']
    else:
        c_id = 0
    if env_id in ["HCWithPosTest-v0",
                  "AntWallTest-v0",
                  "HCWithPos-v0",
                  "AntWall-v0",
                  ]:
        if c_id == 0:
            return partial(wall_behind, -3)
        elif c_id == 1:
            return partial(wall_behind, 3)
    elif env_id in ["SwimmerWithPos-v0",
                    "SwimmerWithPosTest-v0"
                    ]:
        if c_id == 0:
            return partial(wall_infront, 0.5)  # -0.1
    elif env_id in ["InvertedPendulumWall-v0",
                    "InvertedPendulumWallTest-v0",
                    ]:
        if c_id == 0:
            return partial(wall_behind, -0.015)
    elif env_id in ["WalkerWithPos-v0",
                    "WalkerWithPosTest-v0", ]:
        if c_id == 0:
            return partial(wall_behind, -3)
    elif env_id in ["WGW-v0"]:
        if c_id == 0:
            unsafe_states = env_configs['unsafe_states']
            return partial(wall_in, unsafe_states)
    elif env_id in ["Circle-v0",]:
        return null_cost
    else:
        print("Cost function for %s is not implemented yet. Returning null cost function" % env_id)
        return null_cost


# ============================================================================
# General cost functions
# ============================================================================

def wall_behind(pos, obs, acs):
    return (obs[..., 0] < pos)


def wall_infront(pos, obs, acs):
    return (obs[..., 0] > pos)


def wall_in(unsafe_states, obs, acs):
    return (obs in unsafe_states)


def wall_behind_and_infront(pos_back, pos_front, obs, acs):
    return (obs[..., 0] <= pos_back).astype(np.float32) + (obs[..., 0] >= pos_front).astype(np.float32)


def null_cost(x, *args):
    # Zero cost everywhere
    return 0


def torque_constraint(threshold, obs, acs):
    return np.any(np.abs(acs) > threshold, axis=-1)