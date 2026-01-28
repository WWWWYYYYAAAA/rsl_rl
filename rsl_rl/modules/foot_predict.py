# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn
import copy
from rsl_rl.networks import MLP, EmpiricalNormalization


class FootPredict(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        activation: str = "elu",
        history_length = 6,
        beta = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        self.num_obs = 270
        self.lin_vel = 3
        self.ang_vel = 3
        self.feet_pos_body = 12
        self.feet_vel_body = 12
        self.elevation_size = 187
        self.pred_in = self.num_obs + (self.lin_vel + self.ang_vel + self.feet_pos_body + self.feet_vel_body)*history_length+self.elevation_size
        self.pred_out = 4+12 # contact flag contact pos
        self.fpred = MLP(self.pred_in, self.pred_out, [1024,512,256], activation)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> None:
        raise NotImplementedError
    
    @property
    def action_mean(self) -> torch.Tensor:
        if hasattr(self, 'distribution') and self.distribution is not None:
            return self.distribution.mean
        else:
            return torch.zeros(self.num_actions)
        # return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            return torch.ones(self.num_actions) * 0.1  # 小标准差
        return self.distribution.stddev
    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            return torch.ones(self.num_actions) * 0.1
        return self.distribution.entropy().sum(dim=-1)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)
    
    def get_extra_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["extra"]]
        return torch.cat(obs_list, dim=-1)
    
    def get_contactflag_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["cf"]]
        return torch.cat(obs_list, dim=-1)
    
    def get_pred_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["pred"]]
        return torch.cat(obs_list, dim=-1)
    
    def get_footpos_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["footpos"]]
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        super().load_state_dict(state_dict, strict=strict)
        return True

