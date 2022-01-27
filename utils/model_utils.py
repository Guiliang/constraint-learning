def get_net_arch(config):
    """
    Returns a dictionary with sizes of layers in policy network,
    value network and cost value network.
    """

    if config['PPO']['cost_vf_layers']:
        separate_layers = dict(pi=config['PPO']['policy_layers'],  # Policy Layers
                               vf=config['PPO']['reward_vf_layers'],  # Value Function Layers
                               cvf=config['PPO']['cost_vf_layers'])  # Cost Value Function Layers
    else:
        # print("Could not define layers for policy, value func and " + \
        #       "cost_value_function, will attempt to just define " + \
        #       "policy and value func")
        separate_layers = dict(pi=config['PPO']['policy_layers'],  # Policy Layers
                               vf=config['PPO']['reward_vf_layers'])  # Value Function Layers

    # if config.shared_layers is not None:
    #     return [*config.shared_layers, separate_layers]
    # else:
    return [separate_layers]
