import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_constraints(cost_function, feature_range, select_dim, obs_dim, acs_dim,
                     save_name, device='cpu', feature_data=None, feature_cost=None, feature_name=None,
                     empirical_input_means=None, num_points=1000, axis_size=24):
    import matplotlib as mpl
    mpl.rcParams['xtick.labelsize'] = axis_size
    mpl.rcParams['ytick.labelsize'] = axis_size
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    selected_feature_generation = np.linspace(feature_range[0], feature_range[1], num_points)
    if empirical_input_means is None:
        input_all = np.zeros((num_points, obs_dim + acs_dim))
    else:
        assert len(empirical_input_means) == obs_dim + acs_dim
        input_all = np.expand_dims(empirical_input_means, 0).repeat(num_points, axis=0)
        # input_all = torch.tensor(input_all)
    input_all[:, select_dim] = selected_feature_generation
    with torch.no_grad():
        obs = input_all[:, :obs_dim]
        acs = input_all[:, obs_dim:]
        preds = cost_function(obs=obs, acs=acs, mode='mean')  # use the mean of a distribution for visualization
    ax[0].plot(selected_feature_generation, preds, c='r', linewidth=5)
    if feature_data is not None:
        ax[0].scatter(feature_data, feature_cost)
        ax[1].hist(feature_data, bins=40, range=(feature_range[0], feature_range[1]))
        ax[1].set_axisbelow(True)
        # Turn on the minor TICKS, which are required for the minor GRID
        ax[1].minorticks_on()
        ax[1].grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax[1].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax[1].set_xlabel(feature_name, fontdict={'fontsize': axis_size})
        ax[1].set_ylabel('Frequency', fontdict={'fontsize': axis_size})
    ax[0].set_xlabel(feature_name, fontdict={'fontsize': axis_size})
    ax[0].set_ylabel('Cost', fontdict={'fontsize': axis_size})
    ax[0].set_ylim([0, 1])
    ax[0].set_xlim(feature_range)
    ax[0].set_axisbelow(True)
    # Turn on the minor TICKS, which are required for the minor GRID
    ax[0].minorticks_on()
    ax[0].grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax[0].grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    fig.savefig(save_name)
    plt.close(fig=fig)
