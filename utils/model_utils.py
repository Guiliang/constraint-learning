import torch
import numpy as np


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


def handle_model_parameters(model, active_keywords, model_name, log_file, set_require_grad):
    """determine which parameters should be fixed"""
    # exclude some parameters from optimizer
    param_frozen_list = []  # should be changed into torch.nn.ParameterList()
    param_active_list = []  # should be changed into torch.nn.ParameterList()
    fixed_parameters_keys = []
    active_parameters_keys = []
    parameters_info = []

    for k, v in model.named_parameters():
        keep_this = False
        size = torch.numel(v)
        parameters_info.append("{0}:{1}".format(k, size))
        for keyword in active_keywords:
            if keyword in k:
                param_active_list.append(v)
                active_parameters_keys.append(k)
                keep_this = True
                break
        if not keep_this:
            param_frozen_list.append(v)
            if set_require_grad:
                v.requires_grad = False  # fix the parameters https://pytorch.org/docs/master/notes/autograd.html
            fixed_parameters_keys.append(k)

    print('-' * 30 + '{0} Optimizer'.format(model_name) + '-' * 30, file=log_file, flush=True)
    print("Active parameters are: {0}".format(str(active_parameters_keys)), file=log_file, flush=True)
    print("Fixed parameters are: {0}".format(str(fixed_parameters_keys)), file=log_file, flush=True)
    # print(parameters_info, file=log_file, flush=True)
    param_frozen_list = torch.nn.ParameterList(param_frozen_list)
    param_active_list = torch.nn.ParameterList(param_active_list)
    print('-' * 60, file=log_file, flush=True)

    return param_frozen_list, param_active_list


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def stability_loss(input_data, aggregates, concepts, relevances):
    """Computes Robustness Loss for the Compas data

    Formulated by Alvarez-Melis & Jaakkola (2018)
    [https://papers.nips.cc/paper/8003-towards-robust-interpretability-with-self-explaining-neural-networks.pdf]
    The loss formulation is specific to the data format
    The concept dimension is always 1 for this project by design
    Parameters
    ----------
    input_data   : torch.tensor
                 Input as (batch_size x num_features)
    aggregates   : torch.tensor
                 Aggregates from SENN as (batch_size x num_classes x concept_dim)
    concepts     : torch.tensor
                 Concepts from Conceptizer as (batch_size x num_concepts x concept_dim)
    relevances   : torch.tensor
                 Relevances from Parameterizer as (batch_size x num_concepts x num_classes)

    Returns
    -------
    robustness_loss  : torch.tensor
        Robustness loss as frobenius norm of (batch_size x num_classes x num_features)
    """
    batch_size = input_data.size(0)
    num_classes = aggregates.size(1)

    grad_tensor = torch.ones(batch_size, num_classes).to(input_data.device)
    J_yx = torch.autograd.grad(outputs=aggregates,
                               inputs=input_data,
                               grad_outputs=grad_tensor,
                               create_graph=True,
                               only_inputs=True)[0]
    # bs x num_features -> bs x num_features x num_classes
    J_yx = J_yx.unsqueeze(-1)

    # J_hx = Identity Matrix; h(x) is identity function
    robustness_loss = J_yx - relevances
    robustness_loss = robustness_loss.norm(p='fro', dim=1)
    return robustness_loss


def dirichlet_kl_divergence_loss(alpha, prior):
    """
    KL divergence between two dirichlet distribution
    The mean is alpha/(alpha+beta) and variance is alpha*beta/(alpha+beta)^2*(alpha+beta+1)
    There are multiple ways of modelling a dirichlet:
    1) by Laplace approximation with logistic normal: https://arxiv.org/pdf/1703.01488.pdf
    2) by directly modelling dirichlet parameters: https://arxiv.org/pdf/1901.02739.pdf
    code reference：
    1） https://github.com/sophieburkhardt/dirichlet-vae-topic-models
    2） https://github.com/is0383kk/Dirichlet-VAE
    """
    analytical_kld = torch.lgamma(torch.sum(alpha, dim=1)) - torch.lgamma(torch.sum(prior, dim=1))
    analytical_kld += torch.sum(torch.lgamma(prior), dim=1)
    analytical_kld -= torch.sum(torch.lgamma(alpha), dim=1)
    minus_term = alpha - prior
    # tmp = torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    digamma_term = torch.digamma(alpha) - \
                   torch.reshape(torch.digamma(torch.sum(alpha, dim=1)), shape=[alpha.shape[0], 1])
    test = torch.sum(torch.mul(minus_term, digamma_term), dim=1)
    analytical_kld += test
    # self.analytical_kld = self.mask * self.analytical_kld  # mask paddings
    return analytical_kld


def torch_kron_prod(a, b):
    """
    :param a: matrix1 of size [b, M]
    :param b: matrix2 of size [b, N]
    :return: matrix of size [b, M, N]
    """
    res = torch.einsum('ij,ik->ijk', [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
    return res
