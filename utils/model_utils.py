import torch


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


def handle_model_parameters(model, fix_keywords, model_name, log_file, set_require_grad):
    """determine which parameters should be fixed"""
    # exclude some parameters from optimizer
    param_frozen_list = []  # should be changed into torch.nn.ParameterList()
    param_active_list = []  # should be changed into torch.nn.ParameterList()
    fixed_parameters_keys = []
    active_parameters_keys = []
    parameters_info = []

    for k, v in model.named_parameters():
        keep_this = True
        size = torch.numel(v)
        parameters_info.append("{0}:{1}".format(k, size))
        for keyword in fix_keywords:
            if keyword in k:
                param_frozen_list.append(v)
                if set_require_grad:
                    v.requires_grad = False  # fix the parameters https://pytorch.org/docs/master/notes/autograd.html
                keep_this = False
                fixed_parameters_keys.append(k)
                break
        if keep_this:
            param_active_list.append(v)
            active_parameters_keys.append(k)
    print('-' * 30 + '{0} Optimizer'.format(model_name) + '-' * 30, file=log_file, flush=True)
    print("Fixed parameters are: {0}".format(str(fixed_parameters_keys)), file=log_file, flush=True)
    print("Active parameters are: {0}".format(str(active_parameters_keys)), file=log_file, flush=True)
    # print(parameters_info, file=log_file, flush=True)
    param_frozen_list = torch.nn.ParameterList(param_frozen_list)
    param_active_list = torch.nn.ParameterList(param_active_list)
    print('-' * 60, file=log_file, flush=True)

    return param_frozen_list, param_active_list