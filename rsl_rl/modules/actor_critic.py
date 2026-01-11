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


class ActorCritic(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        VAE_enable: bool = True,
        VAE_latent_dim = 16,
        VAE_latent_estimate_dim = 3,
        VAE_encoder_hiden_dim = [512,256,128],
        VAE_decoder_hiden_dim = [128,256,512],
        history_length = 6,
        beta = 1.0,
        # device = "cuda:0",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.state_dependent_std = state_dependent_std

        # Actor
        #vae
        self.VAE_latent_dim = VAE_latent_dim #16
        self.VAE_latent_estimate_dim = VAE_latent_estimate_dim #3 velocity
        self.VAE_enable = VAE_enable
        self.obs_one_step_num = int(num_actor_obs/history_length)
        self.beta = beta
        if self.VAE_enable:
            self.estimator = MLP(num_actor_obs, (self.VAE_latent_dim+self.VAE_latent_estimate_dim), VAE_encoder_hiden_dim, activation)
            self.decoder = MLP(self.VAE_latent_dim+self.VAE_latent_estimate_dim, int(num_actor_obs/history_length), VAE_decoder_hiden_dim, activation)
            self.actor =  MLP(int(num_actor_obs/history_length)+self.VAE_latent_dim+self.VAE_latent_estimate_dim, num_actions, actor_hidden_dims, activation)
            self.num_actions = num_actions
            # self.actor = AssembleActor(activation)
            print(f"VAE Estimator MLP: {self.estimator}")
            print(f"VAE Decoder MLP: {self.decoder}")
        else:
            if self.state_dependent_std:
                self.actor = MLP(num_actor_obs, [2, num_actions], actor_hidden_dims, activation)
            else:
                self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        # Critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # Critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

        # self.one_step_idx = [15, 16, 17, 33, 34, 35, 51, 52, 53,
        #                      114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
        #                      186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
        #                      258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]
        # self.idx_offset = [3,3,3, 6,6,6, 9,9,9,
        #                    21,21,21,21,21,21,21,21,21,21,21,21,
        #                    33,33,33,33,33,33,33,33,33,33,33,33,
        #                    45,45,45,45,45,45,45,45,45,45,45,45]
        # self amp
        # self.device=device
        self.self_one_amp_obs_num = 76
        self.self_one_amp_obs_num_half = 44
        self.foot_num = 4
        
        self.left_idxs = [3, 4, 5, 9, 10, 11]
        self.right_idxs = [0, 1, 2, 6, 7, 8]
        self.left_feet_idxs = [1,3]
        self.right_feet_idxs = [0,2]
        self.root_state_num = 12
        self.obs_lin_num = 24
        self.obs_ang_num = 36
        self.self_amp_obs_history_length = 2
        self.amp_lumbda = 10.0
        self.amp_reward_coef = 2.0
        self.use_self_amp = True
        if self.use_self_amp:
            input_dimension = self.self_one_amp_obs_num_half * self.self_amp_obs_history_length
            self.left_discriminator = MLP(input_dimension, 1, [512,256,128], activation)
            self.right_discriminator = MLP(input_dimension, 1, [512,256,128], activation)

            self.self_amp_obs_normalizer = EmpiricalNormalization(self.self_one_amp_obs_num*self.self_amp_obs_history_length)

        

        


    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    # def forward(self) -> NoReturn:
    #     raise NotImplementedError
    def forward(self) -> None:
        raise NotImplementedError

    # def forward(self, obs):
    #     obs = self.actor_obs_normalizer(obs)
    #     if self.VAE_enable:
    #         return self.dwaq_inference(obs)
    #     else:
    #         if self.state_dependent_std:
    #             return self.actor(obs)[..., 0, :]
    #         else:
    #             return self.actor(obs)

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

    def _update_distribution(self, obs: TensorDict):
        if self.VAE_enable:
            mean, latent = self.dwaq_learn(obs)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            self.distribution = Normal(mean, std)
            return latent
        else:
            if self.state_dependent_std:
                # Compute mean and standard deviation
                mean_and_std = self.actor(obs)
                if self.noise_std_type == "scalar":
                    mean, std = torch.unbind(mean_and_std, dim=-2)
                elif self.noise_std_type == "log":
                    mean, log_std = torch.unbind(mean_and_std, dim=-2)
                    std = torch.exp(log_std)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
            else:
                # Compute mean
                mean = self.actor(obs)
                # Compute standard deviation
                if self.noise_std_type == "scalar":
                    std = self.std.expand_as(mean)
                elif self.noise_std_type == "log":
                    std = torch.exp(self.log_std).expand_as(mean)
                else:
                    raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
            self.distribution = Normal(mean, std)

    # def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
    #     obs = self.get_actor_obs(obs)
    #     obs = self.actor_obs_normalizer(obs)
    #     self._update_distribution(obs)
    #     return self.distribution.sample()

    # def act_inference(self, obs: TensorDict) -> torch.Tensor:
    #     obs = self.get_actor_obs(obs)
    #     obs = self.actor_obs_normalizer(obs)
    #     if self.state_dependent_std:
    #         return self.actor(obs)[..., 0, :]
    #     else:
    #         return self.actor(obs)

    def dwaq_inference(self, obs) -> torch.Tensor:
        latent = self.estimator(obs)
        # lantent = lantent[...,:self.VAE_latent_dim+self.VAE_latent_estimate_dim]
        return self.actor(torch.cat((obs[...,-self.obs_one_step_num:],latent), dim=-1))
        # return self.actor(torch.cat((obs[...,self.one_step_idx],lantent), dim=-1))
    
    def dwaq_learn(self, obs):
        latent = self.estimator(obs)
        # lantent_u = lantent[...,:self.VAE_latent_dim+self.VAE_latent_estimate_dim]
        # latent = torch.concatenate((latent[..., :self.VAE_latent_dim],latent[...,self.VAE_latent_dim:].detach()), dim=-1)
        return self.actor(torch.cat((obs[...,-self.obs_one_step_num:],latent), dim=-1)), latent
        # return self.actor(torch.cat((obs[...,self.one_step_idx],lantent_u), dim=-1)), lantent


    def act(self, obs: TensorDict, **kwargs: dict[str, Any]):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.VAE_enable:
            latent = self._update_distribution(obs)
            return self.distribution.sample(), latent
        else:
            self._update_distribution(obs)
            return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        if self.VAE_enable:
            return self.dwaq_inference(obs)
        else:
            if self.state_dependent_std:
                return self.actor(obs)[..., 0, :]
            else:
                return self.actor(obs)
            
    def vae_loss(self, obs, next_obs, latent):
        vel = self.get_critic_obs(obs)[...,:3]
        # print(self.get_actor_obs(obs).shape)
        next_step_obs = self.get_actor_obs(next_obs)[...,-self.obs_one_step_num:]
        # print(next_step_obs[0,:])
        # next_step_obs = self.get_actor_obs(next_obs)[...,self.one_step_idx]
        # latent_u = latent[...,:self.VAE_latent_dim+self.VAE_latent_estimate_dim]
        #whole latentInput
        latent_detach = torch.concatenate((latent[..., :self.VAE_latent_dim],latent[...,self.VAE_latent_dim:].detach()), dim=-1)
        next_step_obs_estimate = self.decoder(latent_detach)
        # latent_var = latent[...,self.VAE_latent_dim+self.VAE_latent_estimate_dim:]
        latent_vel_u = latent[...,self.VAE_latent_dim:]
        # latent_vel_var = latent_var[...,self.VAE_latent_dim:]
        # latent_u = latent[...,:self.VAE_latent_dim]
        # latent_var = latent_var[...,:self.VAE_latent_dim]
        # autoenc_loss = (nn.MSELoss()(code_vel,vel_target) + nn.MSELoss()(decode,decode_target) + beta*(-0.5 * torch.sum(1 + logvar_latent - mean_latent.pow(2) - logvar_latent.exp())))/self.num_mini_batches
        vel_loss = nn.MSELoss()(latent_vel_u, vel)
        rec_loss = nn.MSELoss()(next_step_obs, next_step_obs_estimate)
        # latent_var = torch.clamp(latent_var, max=10.0, min=-20)
        # kl_loss = -0.5 * torch.mean(1 + latent_var - latent_u*latent_u - torch.exp(latent_var))
        return vel_loss*10.0 + rec_loss, vel_loss, rec_loss
        

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
    
    def get_selfamp_obs(self, obs: TensorDict) -> torch.Tensor:
        # print("##########", obs)
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["selfamp"]]
        return torch.cat(obs_list, dim=-1)
    
    def split_selfamp_obs(self, selfamp_obs, device) -> tuple[torch.Tensor, torch.Tensor]:
        # obs_n = selfamp_obs[:,self.self_one_amp_obs_num:]
        # obs_l = selfamp_obs[:,:self.self_one_amp_obs_num]
        self.dir_ang_mask = torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,], requires_grad=True, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        self.dir_lin_mask = torch.tensor([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0,], requires_grad=True, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        obs = selfamp_obs.view(-1, self.self_amp_obs_history_length,self.self_one_amp_obs_num)
        # print(obs.shape)
        foot = obs[:,:, -self.foot_num:]
        left_foot = foot[:,:, self.left_feet_idxs].view(-1, self.self_amp_obs_history_length, int(self.foot_num/2))
        right_foot = foot[:,:, self.right_feet_idxs].view(-1, self.self_amp_obs_history_length, int(self.foot_num/2))
        
        root = obs[:, :, :self.root_state_num].view(-1, self.self_amp_obs_history_length, self.root_state_num)
        # print(self.root_state_num,-self.foot_num-self.obs_lin_num)
        # print(obs[:, :,self.root_state_num:-self.foot_num-self.obs_lin_num].shape)
        obs_ang = obs[:, :,self.root_state_num:-self.foot_num-self.obs_lin_num].view(-1, self.self_amp_obs_history_length, int(self.obs_ang_num/12), 12)
        obs_ang = obs_ang * self.dir_ang_mask

        obs_lin = obs[:, :, -self.foot_num-self.obs_lin_num:-self.foot_num].view(-1, self.self_amp_obs_history_length, int(self.obs_lin_num/12), 12)
        obs_lin = obs_lin * self.dir_lin_mask

        left_obs_ang = obs_ang[:, :, :, self.left_idxs].view(-1, self.self_amp_obs_history_length, int(self.obs_ang_num/2))
        right_obs_ang = obs_ang[:, :, :, self.right_idxs].view(-1, self.self_amp_obs_history_length, int(self.obs_ang_num/2))

        left_obs_lin = obs_lin[:, :, :, self.left_idxs].view(-1, self.self_amp_obs_history_length, int(self.obs_lin_num/2))
        right_obs_lin = obs_lin[:, :, :, self.right_idxs].view(-1, self.self_amp_obs_history_length, int(self.obs_lin_num/2))

        left_obs = torch.cat((root, left_obs_ang, left_obs_lin, left_foot), dim=-1).view(-1, int(self.self_amp_obs_history_length*self.self_one_amp_obs_num_half))
        right_obs = torch.cat((root, right_obs_ang, right_obs_lin, right_foot), dim=-1).view(-1, int(self.self_amp_obs_history_length*self.self_one_amp_obs_num_half))

        return left_obs, right_obs
    
    def self_amp_loss(self, left_obs, right_obs):

        left2left = self.left_discriminator(left_obs)
        right2left = self.left_discriminator(right_obs)

        right2right = self.right_discriminator(right_obs)
        left2right = self.right_discriminator(left_obs)

        left_motion_gradient = torch.autograd.grad(
            outputs=left2left,  
            inputs=left_obs, 
            grad_outputs=torch.ones_like(left2left),  
            create_graph=True,  
            retain_graph=True,  
            only_inputs=True
        )[0]

        right_motion_gradient = torch.autograd.grad(
            outputs=right2right,
            inputs=right_obs,
            grad_outputs=torch.ones_like(right2right),
            create_graph=True,  
            retain_graph=True,  
        )[0]

        
        one_t = torch.ones_like(left2left)
        zero_t = torch.zeros_like(left_motion_gradient)
        left_d_loss = 0.5 * (nn.MSELoss()(left2left,one_t)+ nn.MSELoss()(right2left,-1.0*one_t)) + nn.MSELoss()(left_motion_gradient, zero_t)*self.amp_lumbda
        right_d_loss = 0.5 * (nn.MSELoss()(right2right,one_t)+ nn.MSELoss()(left2right,-1.0*one_t)) + nn.MSELoss()(right_motion_gradient, zero_t)*self.amp_lumbda

        amp_loss = (left_d_loss + right_d_loss)*0.5

        

        return amp_loss, left_d_loss, right_d_loss
    
    def self_amp_reward(self, obs: TensorDict, device):
        selfamp_obs = self.get_selfamp_obs(obs)
        self.self_amp_obs_normalizer.update(selfamp_obs)
        selfamp_obs = self.self_amp_obs_normalizer(selfamp_obs)

        left_obs, right_obs = self.split_selfamp_obs(selfamp_obs, device)

        right2left = self.left_discriminator(right_obs)
        left2right = self.right_discriminator(left_obs)

        amp_reward = self.amp_reward_coef * (torch.clamp(1 - (1/4) * torch.square(right2left - 1), min=0)\
                                             + torch.clamp(1 - (1/4) * torch.square(left2right - 1), min=0))
        return amp_reward.mean(dim=-1)
    
    def self_amp_train(self, obs: TensorDict, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        selfamp_obs = self.get_selfamp_obs(obs)
        left_obs, right_obs = self.split_selfamp_obs(selfamp_obs, device)
        amp_loss, left_d_loss, right_d_loss = self.self_amp_loss(left_obs, right_obs)
        return amp_loss, left_d_loss, right_d_loss




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

