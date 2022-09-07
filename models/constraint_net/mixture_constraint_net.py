import os
import random
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union, List
import numpy as np
import torch
import torch.nn.functional as F
from models.constraint_net.constraint_net import ConstraintNet
from models.nf_net.masked_autoregressive_flow import MADE, BatchNormFlow, Reverse, FlowSequential
from cirl_stable_baselines3.common.torch_layers import create_mlp
from cirl_stable_baselines3.common.utils import update_learning_rate
from torch import nn
from tqdm import tqdm

from utils.data_utils import build_rnn_input
from utils.model_utils import build_code


class MixtureConstraintNet(ConstraintNet):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            latent_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            expert_obs: np.ndarray,
            expert_acs: np.ndarray,
            is_discrete: bool,
            task: str = 'ICRL',
            regularizer_coeff: float = 0.,
            obs_select_dim: Optional[Tuple[int, ...]] = None,
            acs_select_dim: Optional[Tuple[int, ...]] = None,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            no_importance_sampling: bool = True,
            per_step_importance_sampling: bool = False,
            clip_obs: Optional[float] = 10.,
            initial_obs_mean: Optional[np.ndarray] = None,
            initial_obs_var: Optional[np.ndarray] = None,
            action_low: Optional[float] = None,
            action_high: Optional[float] = None,
            target_kl_old_new: float = -1,
            target_kl_new_old: float = -1,
            train_gail_lambda: Optional[bool] = False,
            max_seq_length: float = 10,
            eps: float = 1e-5,
            eta: float = 0.1,
            device: str = "cpu",
            log_file=None,
    ):
        super(MixtureConstraintNet, self).__init__(obs_dim=obs_dim,
                                                   acs_dim=acs_dim,
                                                   hidden_sizes=hidden_sizes,
                                                   batch_size=batch_size,
                                                   lr_schedule=lr_schedule,
                                                   expert_obs=expert_obs,
                                                   expert_acs=expert_acs,
                                                   is_discrete=is_discrete,
                                                   task=task,
                                                   regularizer_coeff=regularizer_coeff,
                                                   obs_select_dim=obs_select_dim,
                                                   acs_select_dim=acs_select_dim,
                                                   optimizer_class=optimizer_class,
                                                   optimizer_kwargs=optimizer_kwargs,
                                                   no_importance_sampling=no_importance_sampling,
                                                   per_step_importance_sampling=per_step_importance_sampling,
                                                   clip_obs=clip_obs,
                                                   initial_obs_mean=initial_obs_mean,
                                                   initial_obs_var=initial_obs_var,
                                                   action_low=action_low,
                                                   action_high=action_high,
                                                   target_kl_old_new=target_kl_old_new,
                                                   target_kl_new_old=target_kl_new_old,
                                                   train_gail_lambda=train_gail_lambda,
                                                   eps=eps,
                                                   device=device,
                                                   log_file=log_file,
                                                   build_net=False)
        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length
        self.eta = eta
        self.pivot_vectors_by_cid = {}
        self._build()

    def _define_input_dims(self) -> None:
        self.input_obs_dim = []
        self.input_acs_dim = []
        if self.obs_select_dim is None:
            self.input_obs_dim += [i for i in range(self.obs_dim)]
        elif self.obs_select_dim[0] != -1:
            self.input_obs_dim += self.obs_select_dim
        obs_len = len(self.input_obs_dim)
        if self.acs_select_dim is None:
            self.input_acs_dim += [i for i in range(self.acs_dim)]
        elif self.acs_select_dim[0] != -1:
            self.input_acs_dim += self.acs_select_dim
        self.select_dim = self.input_obs_dim + [i + obs_len for i in self.input_acs_dim]
        self.input_dim = len(self.select_dim)
        assert self.input_dim > 0, ""

    def _build(self) -> None:

        # Create constraint function and add sigmoid at the end
        self.constraint_functions = [nn.Sequential(
            *create_mlp(self.input_dim, 1, list(self.hidden_sizes)),
            nn.Sigmoid()
        ).to(self.device) for i in range(self.latent_dim)]

        # Creat density model
        modules = []
        for i in range(len(self.hidden_sizes)):
            modules += [
                MADE(num_inputs=self.input_dim,
                     num_hidden=self.hidden_sizes[i],
                     num_cond_inputs=self.latent_dim,
                     ),
                BatchNormFlow(self.input_dim, ),
                Reverse(self.input_dim, )
            ]
        model = FlowSequential(*modules)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)
        model.to(self.device)
        self.density_model = model

        # Build optimizer
        if self.optimizer_class is not None:
            self.cns_optimizers = []
            for i in range(self.latent_dim):
                self.cns_optimizers.append(self.optimizer_class(params=self.constraint_functions[i].parameters(),
                                                                lr=self.lr_schedule(1),
                                                                **self.optimizer_kwargs))
            self.density_optimizer = self.optimizer_class(params=self.density_model.parameters(),
                                                          lr=self.lr_schedule(1),
                                                          **self.optimizer_kwargs)
        else:
            self.cns_optimizers = None
        if self.train_gail_lambda:
            self.criterion = nn.BCELoss()

    def forward(self, x: torch.tensor) -> torch.tensor:
        data = x[:, :-self.latent_dim]
        codes = x[:, -self.latent_dim:]
        outputs = []
        for i in range(self.latent_dim):
            outputs.append(self.constraint_functions[i](data).squeeze(dim=1))
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs * codes
        # tmp = torch.sum(outputs, dim=1)
        return torch.sum(outputs, dim=1)

    def cost_function_with_code(self, obs: np.ndarray, acs: np.ndarray, codes: np.ndarray) -> np.ndarray:
        assert obs.shape[-1] == self.obs_dim, ""
        if not self.is_discrete:
            assert acs.shape[-1] == self.acs_dim, ""

        data = self.prepare_data(obs, acs)
        codes = torch.tensor(codes, dtype=torch.float32).to(self.device)
        function_input = torch.cat([data, codes], dim=1)
        with torch.no_grad():
            out = self.__call__(function_input)
        cost = 1 - out.detach().cpu().numpy()
        return cost

    def call_forward(self, x: np.ndarray):
        with torch.no_grad():
            out = self.__call__(torch.tensor(x, dtype=torch.float32).to(self.device),
                                )
        return out

    def train_nn(
            self,
            iterations: np.ndarray,
            nominal_obs: np.ndarray,
            nominal_acs: np.ndarray,
            episode_lengths: np.ndarray,
            obs_mean: Optional[np.ndarray] = None,
            obs_var: Optional[np.ndarray] = None,
            current_progress_remaining: float = 1,
            **other_parameters,
    ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        # Update normalization stats
        self.current_obs_mean = obs_mean
        self.current_obs_var = obs_var

        # Prepare data
        nominal_data_games = [self.prepare_data(nominal_obs[i], nominal_acs[i]) for i in range(len(nominal_obs))]
        expert_data_games = [self.prepare_data(self.expert_obs[i], self.expert_acs[i]) for i in
                             range(len(self.expert_obs))]

        nominal_data = torch.cat(nominal_data_games, dim=0)
        expert_data = torch.cat(expert_data_games, dim=0)

        assert 'nominal_codes' in other_parameters
        nominal_code_games = [torch.tensor(other_parameters['nominal_codes'][i]).to(self.device) for i in
                              range(len(other_parameters['nominal_codes']))]
        nominal_code = torch.cat(nominal_code_games, dim=0)

        # list all possible candidate codes with shape [batch_size, latent_dim, latent_dim],
        # for example, for latent_dim =2, we have [[1,0],[0,1]]*batch_size
        expert_candidate_code = build_code(code_axis=[i for i in range(self.latent_dim)],
                                           code_dim=self.latent_dim,
                                           num_envs=self.latent_dim)
        expert_candidate_code = torch.tensor(expert_candidate_code).to(self.device)
        for itr in tqdm(range(iterations)):
            for nom_batch_indices, exp_batch_indices in self.get(nominal_data.shape[0], expert_data.shape[0]):
                nominal_data_batch = nominal_data[nom_batch_indices]
                nominal_code_batch = nominal_code[nom_batch_indices]
                m_loss, log_prob = self.density_model.log_probs(inputs=nominal_data_batch,
                                                                cond_inputs=nominal_code_batch)
                self.density_optimizer.zero_grad()
                density_loss = -m_loss.mean()
                log_prob = log_prob.mean()
                density_loss.backward()
                self.density_optimizer.step()

        # save training data for different codes
        nominal_data_by_cid = {}
        for cid in range(self.latent_dim):
            nominal_data_by_cid.update({cid: None})
        nominal_log_prob_by_cid = {}
        for cid in range(self.latent_dim):
            nominal_log_prob_by_cid.update({cid: None})
        expert_data_by_cid = {}
        for cid in range(self.latent_dim):
            expert_data_by_cid.update({cid: None})

        # scan through the expert data and classify them
        expert_code_games = []
        expert_latent_prob_games = []
        for i in range(len(expert_data_games)):
            expert_data_game_repeat = expert_data_games[i].unsqueeze(dim=1).repeat(1, self.latent_dim, 1)
            expert_code_game = expert_candidate_code.unsqueeze(dim=0).repeat(len(expert_data_games[i]), 1, 1)
            _, expert_log_prob = self.density_model.log_probs(
                inputs=expert_data_game_repeat.reshape([-1, self.input_dim]),
                cond_inputs=expert_code_game.reshape(
                    [-1, self.latent_dim]))
            # sum up the log-prob for datapoints to determine the cid of entire trajectory.
            expert_log_sum_game = expert_log_prob.reshape([-1, self.latent_dim, 1]).sum(dim=0).squeeze(dim=-1)
            expert_cid = expert_log_sum_game.argmax()
            print("expert game: {0}, cid: {1}".format(i, expert_cid))
            if expert_data_by_cid[expert_cid.item()] is None:
                expert_data_by_cid[expert_cid.item()] = expert_data_games[i]
            else:
                expert_data_by_cid[expert_cid.item()] = torch.cat([expert_data_by_cid[expert_cid.item()],
                                                                   expert_data_games[i]], dim=0)
            expert_cid_game = expert_cid.repeat(len(expert_data_games[i]))
            # repeat the cid to label all the expert datapoints.
            expert_code_game = F.one_hot(expert_cid_game, num_classes=self.latent_dim).to(self.device)
            expert_code_games.append(expert_code_game)
        expert_codes = torch.cat(expert_code_games, dim=0)

        # scan through the nominal data and pick some pivot points
        for i in range(len(nominal_data_games)):
            nominal_data_game = nominal_data_games[i]
            nominal_code_game = nominal_code_games[i]
            nominal_cid = nominal_code_game[0].argmax()
            if nominal_data_by_cid[nominal_cid.item()] is None:
                nominal_data_by_cid[nominal_cid.item()] = nominal_data_game
            else:
                nominal_data_by_cid[nominal_cid.item()] = torch.cat([nominal_data_by_cid[nominal_cid.item()],
                                                                     nominal_data_game], dim=0)
            nominal_code_game_reverse = torch.ones(size=nominal_code_game.shape)
            _, reverse_log_prob_game = self.density_model.log_probs(inputs=nominal_data_game,
                                                                    cond_inputs=nominal_code_game_reverse)
            if nominal_log_prob_by_cid[nominal_cid.item()] is None:
                nominal_log_prob_by_cid[nominal_cid.item()] = reverse_log_prob_game
            else:
                nominal_log_prob_by_cid[nominal_cid.item()] = torch.cat([nominal_log_prob_by_cid[nominal_cid.item()],
                                                                         reverse_log_prob_game], dim=0)

        for cid in range(self.latent_dim):
            reverse_log_prob_cid = nominal_log_prob_by_cid[cid]
            _, botk = (-reverse_log_prob_cid.squeeze(dim=-1)).topk(10, dim=0, largest=True, sorted=False)
            pivot_points = nominal_data_by_cid[cid][botk]
            self.pivot_vectors_by_cid.update({cid: pivot_points.mean(dim=0)})
            print('cid: {0}, pivot_vectors is {1}'.format(cid, self.pivot_vectors_by_cid[cid]),
                  flush=True, file=self.log_file)
            for itr in tqdm(range(iterations)):
                # Do a complete pass on data
                for nom_batch_indices, exp_batch_indices in self.get(nominal_data_by_cid[cid].shape[0],
                                                                     expert_data_by_cid[cid].shape[0]):
                    # Get batch data
                    nominal_data_batch = nominal_data_by_cid[cid][nom_batch_indices]
                    expert_data_batch = expert_data_by_cid[cid][exp_batch_indices]
                    # Make predictions
                    nom_cid_code = build_code(code_axis=[cid for i in range(len(nom_batch_indices))],
                                              code_dim=self.latent_dim,
                                              num_envs=len(nom_batch_indices))
                    nom_cid_code = torch.tensor(nom_cid_code).to(self.device)
                    nominal_preds = self.__call__(torch.cat([nominal_data_batch, nom_cid_code], dim=1))
                    expert_cid_code = build_code(code_axis=[cid for i in range(len(exp_batch_indices))],
                                                 code_dim=self.latent_dim,
                                                 num_envs=len(exp_batch_indices))
                    expert_cid_code = torch.tensor(expert_cid_code).to(self.device)
                    expert_preds = self.__call__(torch.cat([expert_data_batch, expert_cid_code], dim=1))

                    # Calculate loss
                    expert_loss = torch.mean(torch.log(expert_preds + self.eps))
                    nominal_loss = torch.mean(torch.log(nominal_preds + self.eps))
                    regularizer_loss = self.regularizer_coeff * (
                            torch.mean(1 - expert_preds) + torch.mean(1 - nominal_preds))
                    discriminator_loss = (-expert_loss + nominal_loss) + regularizer_loss

                    # Update
                    self.cns_optimizers[cid].zero_grad()
                    discriminator_loss.backward()
                    self.cns_optimizers[cid].step()

        bw_metrics = {"backward/cn_loss": discriminator_loss.item(),
                      "backward/density_loss": density_loss.item(),
                      "backward/expert_loss": expert_loss.item(),
                      "backward/nominal_loss": nominal_loss.item(),
                      "backward/regularizer_loss": regularizer_loss.item(),
                      "backward/nominal_preds_max": torch.max(nominal_preds).item(),
                      "backward/nominal_preds_min": torch.min(nominal_preds).item(),
                      "backward/nominal_preds_mean": torch.mean(nominal_preds).item(),
                      "backward/expert_preds_max": torch.max(expert_preds).item(),
                      "backward/expert_preds_min": torch.min(expert_preds).item(),
                      "backward/expert_preds_mean": torch.mean(expert_preds).item(),
                      }
        return bw_metrics

    def save(self, save_path):
        state_dict = dict(
            cn_network=[self.constraint_functions[i].state_dict() for i in range(self.latent_dim)],
            density_model=self.density_model.state_dict(),
            cn_optimizers=[self.cns_optimizers[i].state_dict() for i in range(self.latent_dim)],
            density_optimizer=self.density_optimizer.state_dict(),
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
            hidden_sizes=self.hidden_sizes,
            latent_dim=self.latent_dim,
            input_dim=self.input_dim,
        )
        torch.save(state_dict, save_path)

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        for i in range(self.latent_dim):
            update_learning_rate(self.cns_optimizers[i], self.lr_schedule(current_progress_remaining))
        update_learning_rate(self.density_optimizer, self.lr_schedule(current_progress_remaining))

    def _load(self, load_path):
        state_dict = torch.load(load_path)
        if "cn_network" in state_dict:
            self.constraint_functions.load_state_dict(dic["cn_network"])
        if "cn_optimizer" in state_dict and self.cns_optimizer is not None:
            self.cns_optimizer.load_state_dict(dic["cn_optimizer"])

    # Provide basic functionality to load this class
    @classmethod
    def load(
            cls,
            load_path: str,
            obs_dim: Optional[int] = None,
            acs_dim: Optional[int] = None,
            latent_dim: Optional[int] = None,
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
        if latent_dim is None:
            latent_dim = state_dict["latent_dim"]

        # Create network
        hidden_sizes = state_dict["hidden_sizes"]
        constraint_net = cls(
            obs_dim=obs_dim,
            acs_dim=acs_dim,
            hidden_sizes=hidden_sizes,
            batch_size=None,
            lr_schedule=None,
            expert_obs=None,
            expert_acs=None,
            optimizer_class=None,
            is_discrete=is_discrete,
            obs_select_dim=obs_select_dim,
            acs_select_dim=acs_select_dim,
            clip_obs=clip_obs,
            initial_obs_mean=obs_mean,
            initial_obs_var=obs_var,
            action_low=action_low,
            action_high=action_high,
            device=device,
            latent_dim=latent_dim
        )
        constraint_net.constraint_functions.load_state_dict(state_dict["cn_network"])

        return constraint_net
