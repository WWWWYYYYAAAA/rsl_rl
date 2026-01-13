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


class Discriminator(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        activation: str = "elu",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs])
            )
        super().__init__()

        # Get the observation dimensions


        # Action distribution
        # Note: Populated in update_distribution
        self.distribution = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)
        # self amp
        # self.device=device
        self.self_one_amp_obs_num = 52
        self.self_one_amp_obs_num_half = 26
        self.foot_num = 4
        
        self.left_idxs = [3, 4, 5, 9, 10, 11]
        self.right_idxs = [0, 1, 2, 6, 7, 8]
        self.left_feet_idxs = [1,3]
        self.right_feet_idxs = [0,2]
        self.root_state_num = 0
        self.obs_lin_num = 12
        self.obs_ang_num = 36
        self.self_amp_obs_history_length = 2
        self.amp_lumbda = 10.0
        self.amp_reward_coef = 0.2
        self.use_self_amp = True
        if self.use_self_amp:
            input_dimension = self.self_one_amp_obs_num_half * self.self_amp_obs_history_length
            self.left_discriminator = MLP(input_dimension, 1, [256,128], activation)
            # self.right_discriminator = MLP(input_dimension, 1, [512,256,128], activation)

            # self.self_amp_obs_normalizer = EmpiricalNormalization(self.self_one_amp_obs_num*self.self_amp_obs_history_length)

    def forward(self) -> None:
        raise NotImplementedError

    def get_selfamp_obs(self, obs: TensorDict) -> torch.Tensor:
        # print("##########", obs)
        obs_list = [obs["selfamp"]]
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
        if self.root_state_num > 0:
            root = obs[:, :, :self.root_state_num].view(-1, self.self_amp_obs_history_length, self.root_state_num)

            left_obs = torch.cat((root, left_obs_ang, left_obs_lin, left_foot), dim=-1).view(-1, int(self.self_amp_obs_history_length*self.self_one_amp_obs_num_half))
            right_obs = torch.cat((root, right_obs_ang, right_obs_lin, right_foot), dim=-1).view(-1, int(self.self_amp_obs_history_length*self.self_one_amp_obs_num_half))

            return left_obs, right_obs
        else:
            left_obs = torch.cat((left_obs_ang, left_obs_lin, left_foot), dim=-1).view(-1, int(self.self_amp_obs_history_length*self.self_one_amp_obs_num_half))
            right_obs = torch.cat((right_obs_ang, right_obs_lin, right_foot), dim=-1).view(-1, int(self.self_amp_obs_history_length*self.self_one_amp_obs_num_half))
            return left_obs, right_obs
    
    def self_amp_loss(self, left_obs, right_obs):

        left2left = self.left_discriminator(left_obs)
        right2left = self.left_discriminator(right_obs)

        # right2right = self.right_discriminator(right_obs)
        # left2right = self.right_discriminator(left_obs)

        left_motion_gradient = torch.autograd.grad(
            outputs=left2left,  
            inputs=left_obs, 
            grad_outputs=torch.ones_like(left2left),  
            create_graph=True,  
            retain_graph=True,  
            only_inputs=True
        )[0]

        # right_motion_gradient = torch.autograd.grad(
        #     outputs=right2right,
        #     inputs=right_obs,
        #     grad_outputs=torch.ones_like(right2right),
        #     create_graph=True,  
        #     retain_graph=True,  
        # )[0]

        
        one_t = torch.ones_like(left2left)
        zero_t = torch.zeros_like(left_motion_gradient)
        left_d_loss = 0.5 * (nn.MSELoss()(left2left,one_t)+ nn.MSELoss()(right2left,-1.0*one_t)) + nn.MSELoss()(left_motion_gradient, zero_t)*self.amp_lumbda
        # right_d_loss = 0.5 * (nn.MSELoss()(right2right,one_t)+ nn.MSELoss()(left2right,-1.0*one_t)) + nn.MSELoss()(right_motion_gradient, zero_t)*self.amp_lumbda

        amp_loss = left_d_loss

        

        return amp_loss, left_d_loss, 0, left2left.mean(dim=0), right2left.mean(dim=0), 0, 0
    
    def self_amp_reward(self, obs: TensorDict, device):
        selfamp_obs = self.get_selfamp_obs(obs)
        # self.self_amp_obs_normalizer.update(selfamp_obs)
        # selfamp_obs = self.self_amp_obs_normalizer(selfamp_obs)

        left_obs, right_obs = self.split_selfamp_obs(selfamp_obs, device)
        with torch.no_grad():
            right2left = self.left_discriminator(right_obs)
            # left2left = self.left_discriminator(left_obs)

        amp_reward = self.amp_reward_coef * (torch.clamp(1 - (1/4) * torch.square(right2left - 1), min=0))#*left2left#\
                                            #  + torch.clamp(1 - (1/4) * torch.square(left2right - 1), min=0))
        # amp_reward = self.amp_reward_coef * right2left * left2left
        # print(amp_reward.shape)
        return amp_reward.squeeze(1).detach()

    def self_amp_train(self, obs: TensorDict, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selfamp_obs = self.get_selfamp_obs(obs)
        left_obs, right_obs = self.split_selfamp_obs(selfamp_obs, device)
        amp_loss, left_d_loss, right_d_loss, left2left, right2left, right2right, left2right = self.self_amp_loss(left_obs, right_obs)
        amp_reward_raw = (torch.clamp(1 - (1/4) * torch.square(right2left - 1), min=0))#\
                                            #  + torch.clamp(1 - (1/4) * torch.square(left2right - 1), min=0))
        # amp_reward_raw = self.amp_reward_coef * right2left * left2left
        return amp_loss, left_d_loss, right_d_loss, left2left, right2left, right2right, left2right, amp_reward_raw.mean(dim=-1)

