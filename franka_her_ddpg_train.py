# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##########################################################################################################
########################################## LAUCH SIMULATION ##############################################
##########################################################################################################
"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num-envs", type=int, default=10, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

##########################################################################################################
############################################# RL SETUP ###################################################
##########################################################################################################
from typing import Dict, Tuple
import torch
import numpy as np
import torch.nn as nn
import gymnasium as gym

from franka_env import ManagerRLGoalEnv, FrankaShadowLiftEnvCfg

from drl.utils.env_utils import IsaacVecEnv
from drl.agent.ddpg import DDPG_GCRL
from drl.memory.her import HERMemory
from drl.learning.her import HER
from drl.utils.optim.adamw import AdamWOptimizer
from drl.utils.nn.seq import SeqGRUNet

ENV_CFG = FrankaShadowLiftEnvCfg()


class DetPolicyNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box
    ):
        super().__init__()
        self.obs_dim = observation_space['observation'].shape[0]
        # For HER: desired_goal is (num_stages, goal_dim), we use the last stage
        if len(observation_space['desired_goal'].shape) == 2:
            self.num_stages, self.goal_dim = observation_space['desired_goal'].shape
        else:
            self.goal_dim = observation_space['desired_goal'].shape[0]
            self.num_stages = 1
        self.act_dim = action_space.shape[0]
        
        self.net = SeqGRUNet(
            obs_dim=self.obs_dim, 
            meta_dim=self.goal_dim,  # Single goal dimension
            out_dim=self.act_dim,
            embed_dim=256, num_layers=1,
            hidden_mlp_dims=[1024, 768, 512],
            use_norm=True, activation='SiLU' 
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        (obs, mask), goal = x['observation'], x['goal']
        
        # If goal has multi-stage dimension, select the final stage
        if goal.dim() == 3:  # (batch, num_stages, goal_dim)
            goal = goal[:, -1, :]  # Take last stage
        elif goal.dim() == 2 and goal.shape[-1] == self.num_stages * self.goal_dim // self.num_stages:
            # Check if it's (batch, num_stages * goal_dim) flattened
            if self.num_stages > 1:
                goal = goal.view(-1, self.num_stages, self.goal_dim)[:, -1, :]
            
        return self.net(obs, goal, mask)
    

class ValueNetwork(nn.Module):
    def __init__(
        self, 
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Box
    ):
        super().__init__()
        self.obs_dim = observation_space['observation'].shape[0]
        # For HER: desired_goal is (num_stages, goal_dim), we use the last stage
        if len(observation_space['desired_goal'].shape) == 2:
            self.num_stages, self.goal_dim = observation_space['desired_goal'].shape
        else:
            self.goal_dim = observation_space['desired_goal'].shape[0]
            self.num_stages = 1
        self.act_dim = action_space.shape[0]
        
        self.net = SeqGRUNet(
            obs_dim=self.obs_dim, 
            meta_dim=self.goal_dim + self.act_dim,  # Single goal dimension + action
            out_dim=1,
            embed_dim=256, num_layers=1,
            hidden_mlp_dims=[1024, 768, 512],
            use_norm=True, activation='SiLU' 
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        (obs, mask), goal, action = x['observation'], x['goal'], x['action']
        
        # If goal has multi-stage dimension, select the final stage
        if goal.dim() == 3:  # (batch, num_stages, goal_dim)
            goal = goal[:, -1, :]  # Take last stage
        elif goal.dim() == 2 and goal.shape[-1] == self.num_stages * self.goal_dim // self.num_stages:
            # Check if it's (batch, num_stages * goal_dim) flattened
            if self.num_stages > 1:
                goal = goal.view(-1, self.num_stages, self.goal_dim)[:, -1, :]
        
        meta = torch.cat([goal, action], dim=-1)
        return self.net(obs, meta, mask)


def make_optimizer_fn(params, model_name: str):
    assert model_name in ['actor', 'critic'], ValueError(model_name)
    return AdamWOptimizer(
        params=params, polyak=5e-3,
        lr=3e-4, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=1e-5
    )
    

if __name__ == '__main__':
    envs = IsaacVecEnv(
        manager=ManagerRLGoalEnv,
        cfg=FrankaShadowLiftEnvCfg,
        num_envs=args_cli.num_envs
    )
    agent = DDPG_GCRL(
        make_policy_fn=DetPolicyNetwork,
        make_value_fn=ValueNetwork,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        make_optimizer_fn=make_optimizer_fn,
        device='cuda'
    )
    memory = HERMemory(
        reward_func=ENV_CFG.reward_func,
        horizon=int(ENV_CFG.episode_length_s / (ENV_CFG.sim.dt * ENV_CFG.decimation)),
        max_length=20000,
        device='cuda'
    )
    learner = HER(
        envs=envs,
        agent=agent,
        memory=memory
    )
    learner.run(
        epochs=2000,
        num_cycles=200,
        num_eval_episodes=100,
        act_rd=0.05,
        num_updates=128,
        batch_size=512,
        future_p=0.8,
        discounted_factor=0.98,
        clip_return=None,
        n_steps=ENV_CFG.num_frames,
        step_decay=0.7
    )