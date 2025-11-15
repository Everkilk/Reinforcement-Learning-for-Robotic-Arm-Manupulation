from typing import Tuple, Union, Callable, Optional, Any, Dict

import torch
import numpy as np
from gymnasium import spaces

from drl.agent.base import Agent
from drl.utils.optim.base import Optimizer

from drl.utils.model.stochastics import TanhNormalPolicy
from drl.utils.model.values import DeepValueFunc

from drl.utils.functional import reduce_by_mean
from drl.utils.general import map_structure, Device, Params

###########################################################################################################
################################### PPO + GOAL-CONDITIONED RL #############################################
###########################################################################################################
class PPO_GCRL(Agent):
    """ Proximal Policy Optimization with Goal-Conditioned Reinforcement Learning. """
    def __init__(
        self, 
        make_policy_fn: Callable[[spaces.Space, spaces.Space], torch.nn.Module],
        make_value_fn: Callable[[spaces.Space, spaces.Space], torch.nn.Module], 
        observation_space: spaces.Dict, 
        action_space: spaces.Box, 
        make_optimizer_fn: Callable[[Params, str], Optimizer],
        log_std_range: Tuple[float, float] = (-6.0, 2.0),
        num_tasks: int = 1,
        clip_ratio: float = 0.2,
        value_clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        device: Device = None
    ):
        # check spaces and parameters
        assert isinstance(observation_space, spaces.Dict), TypeError(f'Invalid observation_space type ({type(observation_space)}).')
        for obs_name in ['observation', 'achieved_goal', 'desired_goal']:
            assert obs_name in observation_space.keys(), KeyError(f'{obs_name} is not in observation_space.')
        assert isinstance(action_space, spaces.Box), TypeError(f'Invalid action_space ({action_space}).')
        
        self.clip_ratio = clip_ratio
        self.value_clip_ratio = value_clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.num_tasks = num_tasks

        # build agent
        super().__init__(
            observation_space=observation_space,
            action_space=action_space, 
            target_policy_path='actor.policy_net',
            models={
                'actor': TanhNormalPolicy(
                    make_policy_fn=make_policy_fn,
                    observation_space=observation_space,
                    action_space=action_space,
                    log_std_range=log_std_range
                ),
                'critic': DeepValueFunc(
                    value_type='v',
                    make_value_fn=make_value_fn,
                    observation_space=observation_space,
                    action_space=action_space,
                    use_target=False
                ),
            },
            coefs=None,
            make_optimizer_fn=make_optimizer_fn,
            device=device    
        )

    # select action
    def forward(self, input_dict: Dict[str, Any], deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.actor.select_best(input_dict)
        return self.actor.sample(input_dict)

    # update models
    def backward(
        self, 
        data: Dict[str, Any], 
        discounted_factor: float = 0.99,
        gae_lambda: float = 0.95,
        clip_return: Union[float, None] = None,  # Accept but ignore (for compatibility)
        **kwargs  # Accept any other keyword arguments from RHER
    ) -> Dict[str, float]:
        # extract data
        observations = data['observations']
        actions = data['actions']
        rewards = data['rewards']  # (batch_size, 1)
        terminals = data['terminals']  # (batch_size, 1)
        goals = data['goals']
        old_log_probs = data['old_log_probs']  # (batch_size, 1)
        old_values = data['old_values']  # (batch_size, 1)
        task_ids = data.get('task_ids', None)

        # compute advantages using GAE
        with torch.no_grad():
            input_dict = {
                'observation': observations,
                'goal': goals
            }
            if task_ids is not None:
                input_dict['task_id'] = task_ids
            
            values = self.critic(input_dict)
            
            # compute returns and advantages
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + discounted_factor * next_value * (1 - terminals[t]) - values[t]
                gae = delta + discounted_factor * gae_lambda * (1 - terminals[t]) * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # update actor (policy)
        input_dict = {
            'observation': observations,
            'goal': goals,
            'action': actions
        }
        if task_ids is not None:
            input_dict['task_id'] = task_ids
        
        _, new_log_probs, entropy = self.actor(input_dict)
        
        # compute ratio and clipped objective
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        
        # policy loss
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()
        
        actor_loss = self.actor_optim.backward(policy_loss + self.entropy_coef * entropy_loss)

        # update critic (value function)
        input_dict = {
            'observation': observations,
            'goal': goals
        }
        if task_ids is not None:
            input_dict['task_id'] = task_ids
        
        new_values = self.critic(input_dict)
        
        # clipped value loss
        value_pred_clipped = old_values + torch.clamp(
            new_values - old_values, -self.value_clip_ratio, self.value_clip_ratio
        )
        value_loss1 = (new_values - returns).pow(2)
        value_loss2 = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        critic_loss = self.critic_optim.backward(self.value_loss_coef * value_loss)

        # return evaluations
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy.mean().item(),
            'approx_kl': ((ratio - 1) - ratio.log()).mean().item(),
            'clip_frac': (torch.abs(ratio - 1) > self.clip_ratio).float().mean().item(),
        }
    
    # estimate values
    @torch.inference_mode()
    def compute_state_value(self, input_dict: Dict[str, Any]) -> torch.Tensor:
        # format data
        input_dict = map_structure(self.format_data, input_dict)
        # compute V-values
        return self.critic(input_dict)
