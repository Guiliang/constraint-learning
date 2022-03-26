import sys
from itertools import accumulate
from typing import Tuple, Callable, Optional, Type, Dict, Any, Union
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from constraint_models.ssicrl.aggregators import SumAggregator
from constraint_models.ssicrl.conceptizers import IdentityConceptizer
from constraint_models.ssicrl.parameterizers import LinearParameterizer
from stable_baselines3.common.torch_layers import create_mlp, ResBlock
from stable_baselines3.common.utils import update_learning_rate
from utils.model_utils import handle_model_parameters, dirichlet_kl_divergence_loss, stability_loss


class TrajectoryNet(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            expert_traj_obs: np.ndarray,
            expert_traj_acs: np.ndarray,
            expert_traj_rs: np.ndarray,
            is_discrete: bool,
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = False,
            per_step_importance_sampling: bool = False,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            target_kl_old_new: float = -1,
            target_kl_new_old: float = -1,
            train_gail_lambda: Optional[bool] = False,
            eps: float = 1e-5,
            dir_prior: float = 1,
            discount_factor: float = 1,
            log_std_init: float = 0.0,
            max_seq_len: int = 300,
            device: str = "cpu",
            log_file=None
    ):
        super(TrajectoryNet, self).__init__()
        self.log_file = log_file
        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.obs_select_dim = obs_select_dim
        self.acs_select_dim = acs_select_dim
        self._define_input_dims()

        self.expert_traj_obs = expert_traj_obs
        self.expert_traj_acs = expert_traj_acs
        self.expert_traj_rs = expert_traj_rs

        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.is_discrete = is_discrete
        self.regularizer_coeff = regularizer_coeff
        self.importance_sampling = not no_importance_sampling
        self.per_step_importance_sampling = per_step_importance_sampling
        self.clip_obs = clip_obs
        self.device = device
        self.eps = eps
        self.dir_prior = dir_prior
        self.discount_factor = discount_factor

        self.train_gail_lambda = train_gail_lambda

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class
        self.lr_schedule = lr_schedule

        self.current_obs_mean = initial_obs_mean
        self.current_obs_var = initial_obs_var
        self.action_low = action_low
        self.action_high = action_high

        self.target_kl_old_new = target_kl_old_new
        self.target_kl_new_old = target_kl_new_old

        self.current_progress_remaining = 1.
        self.log_std_init = log_std_init
        self.max_seq_len = max_seq_len

        self._build()

    def _define_input_dims(self) -> None:
        self.select_obs_dim = []
        self.select_acs_dim = []
        if self.obs_select_dim is None:
            self.select_obs_dim += [i for i in range(self.obs_dim)]
        elif self.obs_select_dim[0] != -1:
            self.select_obs_dim += self.obs_select_dim
        if self.acs_select_dim is None:
            self.select_acs_dim += [i for i in range(self.acs_dim)]
        elif self.acs_select_dim[0] != -1:
            self.select_acs_dim += self.acs_select_dim
        self.input_dims = len(self.select_obs_dim) + len(self.select_acs_dim)
        assert self.input_dims > 0, ""

    def _build(self) -> None:
        """
        build the network, what we need
        1) model the parameters of the constraint distribution following the beta distribution.
        2) model the Q function for representing the constraints.
        3) model the policy for approximating the policy represented by the Q function.
        :return:
        """
        self.sample_conceptizer = IdentityConceptizer().to(self.device)
        self.sample_parameterizer = LinearParameterizer(num_concepts=self.input_dims,
                                                        num_classes=2,
                                                        hidden_sizes=[self.input_dims] +
                                                                     self.hidden_sizes +
                                                                     [2 * self.input_dims]
                                                        ).to(self.device)
        self.sample_aggregator = SumAggregator(num_classes=1).to(self.device)

        self.traj_encoder = ResBlock(input_dims=self.input_dims).to(self.device)
        self.traj_conceptizer = IdentityConceptizer().to(self.device)
        self.traj_parameterizer = LinearParameterizer(num_concepts=self.max_seq_len,
                                                      num_classes=2,
                                                      hidden_sizes=[self.input_dims] +
                                                                   self.hidden_sizes +
                                                                   [2 * self.input_dims]
                                                      ).to(self.device)
        #
        # # predict the Q(s,a) values, the action is continuous
        # self.q_net = nn.Sequential(
        #     *create_mlp(self.input_dims, 1, self.hidden_sizes)
        # )
        # self.q_net.to(self.device)
        #
        # # predict the policy \pi(a|s) values, the action is continuous. Actor-Critic should be employed
        # self.policy_network = nn.Sequential(
        #     *create_mlp(self.obs_dim, self.acs_dim, self.hidden_sizes)
        # )
        # self.log_std = nn.Parameter(torch.ones(self.acs_dim) * self.log_std_init, requires_grad=True).to(self.device)
        # self.policy_network.to(self.device)
        #
        # build different optimizers for different models
        self.optimizers = {'ICRL': None, 'policy': None}
        param_active_key_words = {'ICRL': ['sample'],
                                  'policy': ['policy_network'], }
        for key in self.optimizers.keys():
            param_frozen_list, param_active_list = \
                handle_model_parameters(model=self,
                                        active_keywords=param_active_key_words[key],
                                        model_name=key,
                                        log_file=self.log_file,
                                        set_require_grad=False)
            if self.optimizer_class is not None:
                optimizer = self.optimizer_class([{'params': param_frozen_list, 'lr': 0.0},
                                                  {'params': param_active_list,
                                                   'lr': self.lr_schedule(1)}],
                                                 lr=self.lr_schedule(1))
            else:
                optimizer = None
            self.optimizers.update({key: optimizer})

    def forward(self, x: torch.tensor) -> torch.tensor:
        alpha_beta = self.traj_encoder(x)
        alpha = alpha_beta[:, 0]
        beta = alpha_beta[:, 1]
        QValues = self.q_net(x)
        policy = self.policy_network(x)

        return alpha, beta, QValues, policy

    def cost_function(self, obs: np.ndarray, acs: np.ndarray) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""
        obs, acs = self.prepare_data(obs, acs)
        cost_input = torch.cat([obs, acs], dim=-1)
        with torch.no_grad():
            alpha, beta, _, _ = self.__call__(cost_input)
        # TODO: Maybe we should not use the expectation of beta distribution
        out = alpha / (alpha + beta)
        cost = 1 - out.detach().cpu().numpy()
        # return cost.squeeze(axis=-1)
        return cost

    def call_forward(self, x: np.ndarray):
        with torch.no_grad():
            out = self.__call__(torch.tensor(x, dtype=torch.float32).to(self.device))
        return out

    def predict(self, obs: torch.tensor, deterministic=False):
        mean_action = self.policy_network(obs)
        std = torch.exp(self.log_std)
        eps = torch.randn_like(std)
        if deterministic:
            action = mean_action
        else:
            action = eps * std + mean_action

        output_action = []
        for i in range(self.acs_dim):
            output_action.append(torch.clamp(action[:, i], min=self.action_low[i], max=self.action_high[i]))

        return torch.stack(output_action, dim=1)

    def train_nn(
            self,
            iterations: int,
            nominal_traj_obs: list,
            nominal_traj_acs: list,
            nominal_traj_rs: list,
            # episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
    ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        assert iterations > 0
        for itr in tqdm(range(iterations)):
            # TODO: maybe we do need the importance sampling
            # is_weights = torch.ones(expert_input.shape[0])

            loss_all_step = []
            expert_stab_loss_all_step = []
            expert_kld_loss_all_step = []
            expert_recon_loss_all_step = []
            expert_preds_all_step = []
            nominal_stab_loss_all_step = []
            nominal_kld_loss_all_step = []
            nominal_recon_loss_all_step = []
            nominal_preds_all_step = []

            # Do a complete pass on data, we don't have to use the get(), but let's leave it here for future usage
            for batch_indices in self.get(len(self.expert_traj_obs), len(nominal_traj_obs)):

                # Get batch data
                batch_expert_traj_obs = [self.expert_traj_obs[i] for i in batch_indices[0]]
                batch_expert_traj_acs = [self.expert_traj_acs[i] for i in batch_indices[0]]
                batch_expert_traj_rs = [self.expert_traj_rs[i] for i in batch_indices[0]]
                batch_nominal_traj_obs = [nominal_traj_obs[i] for i in batch_indices[1]]
                batch_nominal_traj_acs = [nominal_traj_acs[i] for i in batch_indices[1]]
                batch_nominal_traj_rs = [nominal_traj_rs[i] for i in batch_indices[1]]
                batch_max_seq_len = max([len(item) for item in batch_expert_traj_obs] +
                                        [len(item) for item in batch_nominal_traj_obs])
                batch_seq_len = min(batch_max_seq_len, self.max_seq_len)

                batch_expert_traj_obs, batch_expert_traj_acs, batch_expert_traj_rs = self.prepare_data(
                    obs=batch_expert_traj_obs,
                    acs=batch_expert_traj_acs,
                    rs=batch_expert_traj_rs,
                    max_seq_len=batch_seq_len
                )

                batch_nominal_traj_obs, batch_nominal_traj_acs, batch_nominal_traj_rs = self.prepare_data(
                    obs=batch_nominal_traj_obs,
                    acs=batch_nominal_traj_acs,
                    rs=batch_nominal_traj_rs,
                    max_seq_len=batch_seq_len
                )

                batch_size = batch_expert_traj_obs.shape[0]
                prior = (torch.ones((batch_size, 2), dtype=torch.float32) * self.dir_prior).to(self.device)
                traj_loss = []
                for i in range(batch_seq_len):
                    expert_stab_loss, expert_kld_loss, expert_recon_loss, expert_preds_t = \
                        self.compute_losses(batch_obs_t=batch_expert_traj_obs[:, i, :],
                                            batch_acs_t=batch_expert_traj_acs[:, i, :],
                                            prior=prior,
                                            expert_loss=True)
                    expert_stab_loss_all_step.append(expert_stab_loss)
                    expert_kld_loss_all_step.append(expert_kld_loss)
                    expert_recon_loss_all_step.append(expert_recon_loss)
                    expert_preds_all_step.append(expert_preds_t)
                    nominal_stab_loss, nominal_kld_loss, nominal_recon_loss, nominal_preds_t = \
                        self.compute_losses(batch_obs_t=batch_nominal_traj_obs[:, i, :],
                                            batch_acs_t=batch_nominal_traj_acs[:, i, :],
                                            prior=prior,
                                            expert_loss=False)
                    nominal_stab_loss_all_step.append(nominal_stab_loss)
                    nominal_kld_loss_all_step.append(nominal_kld_loss)
                    nominal_recon_loss_all_step.append(nominal_recon_loss)
                    nominal_preds_all_step.append(nominal_preds_t)
                    loss_t = (expert_recon_loss + nominal_recon_loss) + \
                             self.regularizer_coeff * (expert_kld_loss + nominal_kld_loss) + \
                             self.regularizer_coeff * (expert_stab_loss + nominal_stab_loss)
                    loss_all_step.append(loss_t)
                    traj_loss.append(loss_t)

                self.optimizers['ICRL'].zero_grad()
                ave_traj_loss = torch.mean(torch.stack(traj_loss))
                ave_traj_loss.backward()
                self.optimizers['ICRL'].step()
            ave_loss_all_step = torch.mean(torch.stack(loss_all_step))
            ave_expert_stab_loss_all_step = torch.mean(torch.stack(expert_stab_loss_all_step))
            ave_expert_kld_loss_all_step = torch.mean(torch.stack(expert_kld_loss_all_step))
            ave_expert_recon_loss_all_step = torch.mean(torch.stack(expert_recon_loss_all_step))
            expert_preds_all_step = torch.cat(expert_preds_all_step, dim=0)
            ave_nominal_stab_loss_all_step = torch.mean(torch.stack(nominal_stab_loss_all_step))
            ave_nominal_kld_loss_all_step = torch.mean(torch.stack(nominal_kld_loss_all_step))
            ave_nominal_recon_loss_all_step = torch.mean(torch.stack(nominal_recon_loss_all_step))
            nominal_preds_all_step = torch.cat(nominal_preds_all_step, dim=0)

        bw_metrics = {"backward/loss": ave_loss_all_step.item(),
                      "backward/expert/stab_loss": ave_expert_stab_loss_all_step.item(),
                      "backward/expert/kld_loss": ave_expert_kld_loss_all_step.item(),
                      "backward/expert/recon_loss": ave_expert_recon_loss_all_step.item(),
                      "backward/nominal/stab_loss": ave_nominal_stab_loss_all_step.item(),
                      "backward/nominal/kld_loss": ave_nominal_kld_loss_all_step.item(),
                      "backward/nominal/recon_loss": ave_nominal_recon_loss_all_step.item(),
                      # "backward/is_mean": torch.mean(is_weights).detach().item(),
                      # "backward/is_max": torch.max(is_weights).detach().item(),
                      # "backward/is_min": torch.min(is_weights).detach().item(),
                      "backward/nominal/preds_max": torch.max(nominal_preds_all_step).item(),
                      "backward/nominal/preds_min": torch.min(nominal_preds_all_step).item(),
                      "backward/nominal/preds_mean": torch.mean(nominal_preds_all_step).item(),
                      "backward/expert/preds_max": torch.max(expert_preds_all_step).item(),
                      "backward/expert/preds_min": torch.min(expert_preds_all_step).item(),
                      "backward/expert/preds_mean": torch.mean(expert_preds_all_step).item(),
                      }
        # if self.importance_sampling:
        #     stop_metrics = {"backward/kl_old_new": kl_old_new.item(),
        #                     "backward/kl_new_old": kl_new_old.item(),
        #                     "backward/early_stop_itr": early_stop_itr}
        #     bw_metrics.update(stop_metrics)

        return bw_metrics

    def compute_losses(self, batch_obs_t, batch_acs_t, prior, expert_loss):
        batch_input_t = torch.cat([batch_obs_t, batch_acs_t], dim=-1)
        batch_input_t.requires_grad_()  # track all operations on x for jacobian calculation
        sample_concepts, _ = self.sample_conceptizer(batch_input_t)
        sample_relevance = self.sample_parameterizer(batch_input_t)
        # Both the alpha and the beta parameters should be greater than 0,
        alpha_beta_t = F.softplus(self.sample_aggregator(sample_concepts, sample_relevance))
        alpha_t = alpha_beta_t[:, 0]
        beta_t = alpha_beta_t[:, 1]
        preds_t = torch.distributions.Beta(alpha_t, beta_t).rsample()
        # Losses
        stab_loss = stability_loss(input_data=batch_input_t,
                                   aggregates=alpha_beta_t,
                                   concepts=sample_concepts,
                                   relevances=sample_relevance)
        analytical_kld_loss = dirichlet_kl_divergence_loss(
            alpha=torch.stack([alpha_t, beta_t], dim=1),
            prior=prior).mean()
        if expert_loss:
            recon_loss = -torch.mean(torch.log(preds_t + self.eps))
        else:
            recon_loss = torch.mean(torch.log(preds_t + self.eps))

        return stab_loss, analytical_kld_loss, recon_loss, preds_t

    def compute_is_weights(self, preds_old: torch.Tensor, preds_new: torch.Tensor,
                           episode_lengths: np.ndarray) -> torch.tensor:
        with torch.no_grad():
            n_episodes = len(episode_lengths)
            cumulative = [0] + list(accumulate(episode_lengths))

            ratio = (preds_new + self.eps) / (preds_old + self.eps)
            prod = [torch.prod(ratio[cumulative[j]:cumulative[j + 1]])
                    for j in range(n_episodes)]
            prod = torch.tensor(prod)
            normed = n_episodes * prod / (torch.sum(prod) + self.eps)

            if self.per_step_importance_sampling:
                is_weights = torch.tensor(ratio / torch.mean(ratio))
            else:
                is_weights = []
                for length, weight in zip(episode_lengths, normed):
                    is_weights += [weight] * length
                is_weights = torch.tensor(is_weights)

            # Compute KL(old, current)
            kl_old_new = torch.mean(-torch.log(prod + self.eps))
            # Compute KL(current, old)
            prod_mean = torch.mean(prod)
            kl_new_old = torch.mean((prod - prod_mean) * torch.log(prod + self.eps) / (prod_mean + self.eps))

        return is_weights.to(self.device), kl_old_new, kl_new_old

    def padding_input(self,
                      input_data: list,
                      length: int,
                      padding_symbol: int) -> np.ndarray:
        input_data_padding = []
        for i in range(len(input_data)):
            padding_length = length - input_data[i].shape[0]
            if padding_length > 0:
                if len(input_data[i].shape) == 2:
                    padding_data = np.ones([padding_length, input_data[i].shape[1]]) * padding_symbol
                elif len(input_data[i].shape) == 1:
                    padding_data = np.ones([padding_length]) * padding_symbol
                input_sample = np.concatenate([input_data[i], padding_data], axis=0)
            else:
                if len(input_data[i].shape) == 2:
                    input_sample = input_data[i][-length:, :]
                elif len(input_data[i].shape) == 1:
                    input_sample = input_data[i][-length:]
            input_data_padding.append(input_sample)
        return np.asarray(input_data_padding)

    def prepare_data(
            self,
            obs: list,
            acs: list,
            rs: list = None,
            max_seq_len: int = None,
            select_dim: bool = True,
    ) -> torch.tensor:
        bs = len(obs)
        max_seq_len = max_seq_len if max_seq_len is not None else self.max_seq_len
        obs = [self.normalize_obs(obs[i], self.current_obs_mean, self.current_obs_var, self.clip_obs)
               for i in range(bs)]
        acs = [self.clip_actions(acs[i], self.action_low, self.action_high) for i in range(bs)]
        obs = self.padding_input(input_data=obs, length=max_seq_len, padding_symbol=0)
        acs = self.padding_input(input_data=acs, length=max_seq_len, padding_symbol=0)
        if rs is not None:
            rs = self.padding_input(input_data=rs, length=max_seq_len, padding_symbol=0)
        acs = self.reshape_actions(acs)
        if select_dim:
            obs = self.select_appropriate_dims(select_dim=self.select_obs_dim, x=obs)
            acs = self.select_appropriate_dims(select_dim=self.select_acs_dim, x=acs)
        if rs is None:
            return torch.tensor(obs, dtype=torch.float32).to(self.device), \
                   torch.tensor(acs, dtype=torch.float32).to(self.device)
        else:
            return torch.tensor(obs, dtype=torch.float32).to(self.device), \
                   torch.tensor(acs, dtype=torch.float32).to(self.device), \
                   torch.tensor(rs, dtype=torch.float32).to(self.device)

    def select_appropriate_dims(self, select_dim: list, x: Union[np.ndarray, torch.tensor]) -> Union[
        np.ndarray, torch.tensor]:
        return x[..., select_dim]

    def normalize_obs(self, obs: np.ndarray, mean: Optional[float] = None, var: Optional[float] = None,
                      clip_obs: Optional[float] = None) -> np.ndarray:
        bs = obs.shape[0]
        obs = np.reshape(obs, newshape=[-1, obs.shape[-1]])
        # tmp = np.reshape(obs, newshape=[bs, -1, obs.shape[-1]])
        if mean is not None and var is not None:
            mean, var = mean[None], var[None]
            obs = (obs - mean) / np.sqrt(var + self.eps)
        if clip_obs is not None:
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        obs = np.reshape(obs, newshape=[bs, -1, obs.shape[-1]])
        return obs.squeeze()

    def reshape_actions(self, acs):
        if self.is_discrete:
            acs_ = acs.astype(int)
            if len(acs.shape) > 1:
                acs_ = np.squeeze(acs_, axis=-1)
            acs = np.zeros([acs.shape[0], self.acs_dim])
            acs[np.arange(acs_.shape[0]), acs_] = 1.

        return acs

    def clip_actions(self, acs: np.ndarray, low: Optional[float] = None, high: Optional[float] = None) -> np.ndarray:
        if high is not None and low is not None:
            acs = np.clip(acs, low, high)

        return acs

    def get(self, expert_size: int, nom_size: int) -> np.ndarray:
        if self.batch_size is None:
            # Return everything, don't create minibatches
            yield np.arange(expert_size), np.arange(nom_size)
        else:
            size = min(nom_size, expert_size)
            expert_indices = np.random.permutation(expert_size)
            # print(expert_indices)
            nom_indices = np.random.permutation(nom_size)

            batch_size = self.batch_size
            start_idx = 0
            while start_idx < size:
                batch_expert_indices = expert_indices[start_idx:start_idx + batch_size]
                batch_nom_indices = nom_indices[start_idx:start_idx + batch_size]
                yield batch_expert_indices, batch_nom_indices
                start_idx += batch_size

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        for optimizer_name in self.optimizers.keys():
            update_learning_rate(self.optimizers[optimizer_name], self.lr_schedule(current_progress_remaining))

    def save(self, save_path):
        state_dict = dict(
            cn_network=self.network.state_dict(),
            cn_optimizer=self.optimizer.state_dict(),
            obs_dim=self.obs_dim,
            acs_dim=self.acs_dim,
            is_discrete=self.is_discrete,
            obs_select_dim=self.obs_select_dim,
            acs_select_dim=self.acs_select_dim,
            clip_obs=self.clip_obs,
            obs_mean=self.current_obs_mean,
            obs_var=self.current_obs_var,
            action_low=self.action_low,
            action_high=self.action_high,
            device=self.device,
            hidden_sizes=self.hidden_sizes
        )
        torch.save(state_dict, save_path)

    def _load(self, load_path):
        state_dict = torch.load(load_path)
        if "cn_network" in state_dict:
            self.network.load_state_dict(dic["cn_network"])
        if "cn_optimizer" in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(dic["cn_optimizer"])

    # Provide basic functionality to load this class
    @classmethod
    def load(
            cls,
            load_path: str,
            obs_dim: Optional[int] = None,
            acs_dim: Optional[int] = None,
            is_discrete: bool = None,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            clip_obs: Optional[float] = None,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            device: str = "auto"
    ):

        state_dict = torch.load(load_path)
        # If value isn't specified, then get from state_dict
        if obs_dim is None:
            obs_dim = state_dict["obs_dim"]
        if acs_dim is None:
            acs_dim = state_dict["acs_dim"]
        if is_discrete is None:
            is_discrete = state_dict["is_discrete"]
        if obs_select_dim is None:
            obs_select_dim = state_dict["obs_select_dim"]
        if acs_select_dim is None:
            acs_select_dim = state_dict["acs_select_dim"]
        if clip_obs is None:
            clip_obs = state_dict["clip_obs"]
        if obs_mean is None:
            obs_mean = state_dict["obs_mean"]
        if obs_var is None:
            obs_var = state_dict["obs_var"]
        if action_low is None:
            action_low = state_dict["action_low"]
        if action_high is None:
            action_high = state_dict["action_high"]
        if device is None:
            device = state_dict["device"]

        # Create network
        hidden_sizes = state_dict["hidden_sizes"]
        constraint_net = cls(
            obs_dim, acs_dim, hidden_sizes, None, None, None, None,
            is_discrete, None, obs_select_dim, acs_select_dim, None,
            None, None, clip_obs, obs_mean, obs_var, action_low, action_high,
            None, None, device
        )
        constraint_net.network.load_state_dict(state_dict["cn_network"])

        return constraint_net
