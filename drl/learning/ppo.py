from typing import Dict, Any, Union, Callable, Tuple, Optional

import sys
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from drl.learning.base import RLFrameWork, AgentT, MemoryBufferT, VectorizedEnvT
from drl.utils.general import (LOGGER, format_time, format_tabulate, map_structure, 
                               put_structure, groupby_structure, nearest_node_value, MeanMetrics)


class PPO_Learning(RLFrameWork):
    """PPO Learning framework with on-policy rollout collection."""
    def __init__(
        self, 
        envs: VectorizedEnvT, 
        agent: AgentT, 
        *,
        compute_metrics: Optional[Callable] = None,
        horizon: int = 50,
        num_stages: int = 1
    ):
        if compute_metrics is None:
            compute_metrics = lambda x: x.get('goal_achieved', 0.0) + (x['eps_reward'] / x['eps_horizon'])
        super().__init__(envs=envs, agent=agent, memory=None, compute_metrics=compute_metrics)
        
        self.horizon = horizon
        self.num_stages = num_stages
        
        # training params
        self._start_epoch: int = 1
        self._time_delta: float = 0.0
        self._best_eval: float = -float('inf')
        self._total_updates: int = 0
        
    # exploration part ----------------------------------------------------
    def _get_stages(self, observations: Any, mt_goals: Any, reward_func: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        # check goal achieved for all stages
        goal_achieveds = torch.stack([
            reward_func(observations, map_structure(lambda x: x[:, stage_id], mt_goals), stage_id)[-1]
            for stage_id in range(self.num_stages)
        ], dim=-1) # (batch_size, num_stages)
        # select current stages
        stages = (goal_achieveds.fliplr().cummax(dim=1).values == 1).sum(dim=1).clamp(0, self.num_stages - 1)
        return stages, goal_achieveds

    def _select_action(self, observations: Any, mt_goals: Any, stages: torch.Tensor, *, r_mix: float = 0.5) -> torch.Tensor:
        assert 0 <= r_mix <= 1, ValueError(f'Invalid r_mix ({r_mix}).')
        batch_size = nearest_node_value(lambda x: len(x), mt_goals)
        device = self.agent.device
        batch_ind = torch.arange(batch_size, device=device)
        # check stage of the process
        # mix the current stage with the next stage for guiding
        mix_masks = torch.rand(batch_size, device=device) < r_mix
        stages[mix_masks] = torch.clamp(stages[mix_masks] + 1, None, self.num_stages - 1)
        # get goals for explorating
        goals = map_structure(lambda x: x[batch_ind, stages], mt_goals)
        # select actions with current policy
        return self.agent({'observation': observations, 'goal': goals, 'task_id': stages}, deterministic=False)

    def select_actions(
        self, 
        observations: Any, 
        mt_goals: Any, 
        *, 
        r_mix: float = 0.5, 
        deterministic: bool = False,
        reward_func: Optional[Callable] = None
    ) -> Union[Tuple[Any, np.ndarray], Any]:
        if not deterministic:
            observations = map_structure(self.agent.format_data, observations)
            mt_goals = map_structure(self.agent.format_data, mt_goals)
            stages, goal_achieveds = self._get_stages(observations, mt_goals, reward_func)
            actions = self._select_action(observations, mt_goals, stages, r_mix=r_mix)
            actions = map_structure(lambda x: x.cpu().numpy(), actions)
            goal_achieveds = goal_achieveds.cpu().numpy().astype(bool)
            return actions, goal_achieveds
        # get final goals and select actions with the policy
        goals = map_structure(lambda x: x[:, -1], mt_goals)
        stages = torch.full((len(goals),), self.num_stages - 1, dtype=torch.long, device=self.agent.device)
        actions = self.agent({'observation': observations, 'goal': goals, 'task_id': stages}, deterministic=True)
        return map_structure(lambda x: x.cpu().numpy(), actions)
    
    # on-policy rollout collection with old_log_probs and old_values
    def collect_rollouts(
        self, 
        r_mix: float = 0.5,
        reward_func: Optional[Callable] = None
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Collect on-policy rollouts with action log probabilities and value estimates."""
        self.envs.train()
        
        # initialize rollout storage
        rollouts = defaultdict(list)
        eps_rewards = np.zeros(self.envs.num_envs, dtype='float32')
        eps_horizons = np.zeros(self.envs.num_envs, dtype='int32')
        
        # run generating
        observations = self.envs.reset()[0]
        for name, value in observations.items():
            rollouts[name].append(value)
        
        for _ in range(self.horizon):
            # prepare tensors for agent
            obs_tensor = map_structure(self.agent.format_data, observations['observation'])
            goal_tensor = map_structure(self.agent.format_data, observations['desired_goal'])
            
            # get stages if using multi-stage goals
            if self.num_stages > 1 and reward_func is not None:
                stages, goal_achieveds = self._get_stages(obs_tensor, goal_tensor, reward_func)
                # select goals based on stages
                batch_size = nearest_node_value(lambda x: len(x), goal_tensor)
                device = self.agent.device
                batch_ind = torch.arange(batch_size, device=device)
                mix_masks = torch.rand(batch_size, device=device) < r_mix
                stages[mix_masks] = torch.clamp(stages[mix_masks] + 1, None, self.num_stages - 1)
                goals = map_structure(lambda x: x[batch_ind, stages], goal_tensor)
            else:
                # single stage - use desired goal directly
                stages = torch.zeros(self.envs.num_envs, dtype=torch.long, device=self.agent.device)
                goals = map_structure(lambda x: x[:, -1] if x.ndim > 1 else x, goal_tensor)
                goal_achieveds = np.zeros((self.envs.num_envs, 1), dtype=bool)
            
            # get action, log_prob, and value from agent
            with torch.no_grad():
                input_dict = {
                    'observation': obs_tensor,
                    'goal': goals,
                    'task_id': stages
                }
                
                # sample action and get log probability using forward pass
                (actions, log_probs), _ = self.agent.actor(input_dict)
                
                # get value estimate
                values = self.agent.critic({'observation': obs_tensor, 'goal': goals, 'task_id': stages})
            
            # convert to numpy
            actions_np = map_structure(lambda x: x.cpu().numpy(), actions)
            log_probs_np = log_probs.cpu().numpy()
            values_np = values.cpu().numpy()
            
            # apply actions to environment
            next_observations, rewards, terminateds, _, infos = self.envs.step(actions_np)
            eps_rewards += rewards
            eps_horizons += 1
            
            # store rollout data
            rollouts['action'].append(actions_np)
            rollouts['reward'].append(rewards)
            rollouts['terminal'].append(terminateds)
            rollouts['old_log_prob'].append(log_probs_np)
            rollouts['old_value'].append(values_np)
            rollouts['stage'].append(stages.cpu().numpy())
            
            for name, value in next_observations.items():
                rollouts[name].append(value)
            
            if self.num_stages > 1:
                rollouts['goal_achieved'].append(goal_achieveds.cpu().numpy().astype(bool) | terminateds.reshape(-1, 1).astype(bool))
            
            observations = next_observations
        
        # format rollouts (num_envs, horizon, ...)
        rollouts = {
            name: map_structure(lambda x: np.swapaxes(x, 0, 1), groupby_structure(rollout, func=np.stack))
            for name, rollout in rollouts.items()
        }
        
        infos = {name: value for name, value in infos.items() if value.dtype != np.object_}
        info = map_structure(lambda x: np.mean(x), {'eps_reward': eps_rewards, 'eps_horizon': eps_horizons, **infos})
        
        return rollouts, info
    
    # update part ---------------------------------------------------------
    def train(
        self, 
        rollouts: Dict[str, Any],
        num_epochs: int = 4,
        batch_size: int = 256,
        discounted_factor: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Dict[str, Any]:
        """Train PPO agent on collected rollouts."""
        # prepare data from rollouts
        num_envs = self.envs.num_envs
        horizon = self.horizon
        
        # flatten rollouts (num_envs * horizon, ...)
        observations = map_structure(lambda x: x[:, :-1].reshape(-1, *x.shape[2:]), rollouts['observation'])
        actions = rollouts['action'].reshape(-1, *rollouts['action'].shape[2:])
        rewards = rollouts['reward'].reshape(-1, 1)
        terminals = rollouts['terminal'].reshape(-1, 1)
        old_log_probs = rollouts['old_log_prob'].reshape(-1, 1)
        old_values = rollouts['old_value'].reshape(-1, 1)
        stages = rollouts['stage'].reshape(-1)
        
        # convert to tensors first
        observations = map_structure(self.agent.format_data, observations)
        actions = self.agent.format_data(actions)
        rewards = self.agent.format_data(rewards)
        terminals = self.agent.format_data(terminals)
        old_log_probs = self.agent.format_data(old_log_probs)
        old_values = self.agent.format_data(old_values)
        stages = self.agent.format_data(stages)
        
        # get goals - need to select the right goal for each stage
        if 'desired_goal' in rollouts:
            # Flatten: (num_envs, horizon, num_stages, goal_dim) -> (num_envs * horizon, num_stages, goal_dim)
            goals_full = map_structure(lambda x: x[:, :-1].reshape(-1, *x.shape[2:]), rollouts['desired_goal'])
            goals_full = map_structure(self.agent.format_data, goals_full)
            # Select goal based on stage for each timestep
            batch_size = num_envs * horizon
            batch_ind = torch.arange(batch_size, device=self.agent.device)
            goals = map_structure(lambda x: x[batch_ind, stages], goals_full)
        else:
            goals = observations  # fallback
        
        # Compute advantages and returns using GAE (across full trajectories)
        with torch.no_grad():
            # Reshape back to trajectory format for GAE computation
            rewards_traj = rewards.view(num_envs, horizon, 1)
            terminals_traj = terminals.view(num_envs, horizon, 1).float()  # Convert to float for arithmetic
            old_values_traj = old_values.view(num_envs, horizon, 1)
            
            # Compute next values (need last observation from rollouts)
            last_obs = map_structure(lambda x: x[:, -1], rollouts['observation'])
            last_stage = rollouts['stage'][:, -1]
            
            last_obs = map_structure(self.agent.format_data, last_obs)
            last_stage_tensor = self.agent.format_data(last_stage)
            
            # Get the goal for the last stage for each environment
            last_goal_full = map_structure(self.agent.format_data, rollouts['desired_goal'][:, -1])  # (num_envs, num_stages, goal_dim)
            batch_ind = torch.arange(num_envs, device=self.agent.device)
            last_goal = map_structure(lambda x: x[batch_ind, last_stage_tensor], last_goal_full)
            
            last_value = self.agent.critic({
                'observation': last_obs,
                'goal': last_goal,
                'task_id': last_stage_tensor
            })
            
            # Compute GAE for each environment trajectory
            advantages_traj = torch.zeros_like(rewards_traj)
            returns_traj = torch.zeros_like(rewards_traj)
            
            for env_id in range(num_envs):
                gae = 0
                for t in reversed(range(horizon)):
                    if t == horizon - 1:
                        next_value = last_value[env_id]
                    else:
                        next_value = old_values_traj[env_id, t + 1]
                    
                    delta = rewards_traj[env_id, t] + discounted_factor * next_value * (1 - terminals_traj[env_id, t]) - old_values_traj[env_id, t]
                    gae = delta + discounted_factor * gae_lambda * (1 - terminals_traj[env_id, t]) * gae
                    advantages_traj[env_id, t] = gae
                    returns_traj[env_id, t] = gae + old_values_traj[env_id, t]
            
            # Flatten back
            advantages = advantages_traj.reshape(-1, 1)
            returns = returns_traj.reshape(-1, 1)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        avg_evals = MeanMetrics()
        
        # multiple epochs over the same data
        for _ in range(num_epochs):
            # shuffle data indices
            indices = torch.randperm(num_envs * horizon)
            
            # mini-batch updates
            for start_idx in range(0, num_envs * horizon, batch_size):
                end_idx = min(start_idx + batch_size, num_envs * horizon)
                batch_indices = indices[start_idx:end_idx]
                
                # create mini-batch
                batch_obs = map_structure(lambda x: x[batch_indices], observations)
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_goals = map_structure(lambda x: x[batch_indices], goals)
                batch_stages = stages[batch_indices]
                
                # Update actor (policy)
                input_dict = {
                    'observation': batch_obs,
                    'goal': batch_goals,
                    'task_id': batch_stages
                }
                
                # Get distribution parameters for the current policy
                mu, log_std = self.agent.actor.policy_net(input_dict)
                std = torch.exp(torch.clamp(log_std, *self.agent.actor.log_std_range))
                
                # Compute log prob of the actions that were taken
                new_log_probs = self.agent.actor.log_prob_action(batch_actions, mu, std)
                
                # Compute entropy
                entropy = 0.5 * (1.0 + torch.log(2 * np.pi * std.pow(2))).sum(dim=-1, keepdim=True)
                
                # Compute ratio and clipped objective
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.agent.clip_ratio, 1 + self.agent.clip_ratio) * batch_advantages
                
                # Policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                
                actor_loss = self.agent.actor_optim.backward(policy_loss + self.agent.entropy_coef * entropy_loss)
                
                # Update critic (value function)
                value_input = {
                    'observation': batch_obs,
                    'goal': batch_goals,
                    'task_id': batch_stages
                }
                
                new_values = self.agent.critic(value_input)
                
                # Clipped value loss
                value_pred_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values, -self.agent.value_clip_ratio, self.agent.value_clip_ratio
                )
                value_loss1 = (new_values - batch_returns).pow(2)
                value_loss2 = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                critic_loss = self.agent.critic_optim.backward(self.agent.value_loss_coef * value_loss)
                
                # Collect metrics
                evals = {
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
                    'entropy': entropy.mean().item(),
                    'approx_kl': ((ratio - 1) - ratio.log()).mean().item(),
                    'clip_frac': (torch.abs(ratio - 1) > self.agent.clip_ratio).float().mean().item(),
                }
                
                avg_evals.update(evals)
        
        return avg_evals
    
    # learning flow -------------------------------------------------------
    def evaluate(self, num_episodes: int = 10, reward_func: Optional[Callable] = None) -> Dict[str, float]:
        self.envs.eval()

        # initialize params for running
        num_envs, horizon = self.envs.num_envs, self.horizon
        eval_info = MeanMetrics()
        eps_rewards = np.zeros(num_envs, dtype='float32')
        eps_horizons = np.zeros(num_envs, dtype='int32')
        count_episodes = 0

        # start evaluation running
        observations, _ = self.envs.reset()
        while True:
            # select the deterministic actions
            actions = self.select_actions(
                observations['observation'], 
                observations['desired_goal'], 
                deterministic=True,
                reward_func=reward_func
            )

            # apply the selected action into the environment and compute episode rewards
            next_observations, rewards, terminateds, truncateds, infos = self.envs.step(actions)
            eps_rewards += rewards
            eps_horizons += 1

            # forward the next state
            observations = next_observations
            env_ids = np.where(terminateds | truncateds | (eps_horizons == horizon))[0]
            if len(env_ids):
                for env_id in env_ids:
                    # reset environment
                    put_structure(observations, env_id, self.envs.single_reset(env_id)[0])
                    # get episode information and compute evaluation metric
                    eps_info = map_structure(lambda x: x[env_id], infos)
                    eps_info = {name: value for name, value in eps_info.items() if value.dtype != np.object_}
                    eps_info.update({
                        'eps_reward': eps_rewards[env_id], 
                        'eps_horizon': eps_horizons[env_id]
                    })
                    eps_info.update({'eval_value': self.compute_metrics(eps_info)})
                    eval_info.update(eps_info)
                    # reset episode-specific counters
                    eps_rewards[env_id], eps_horizons[env_id] = 0.0, 0
                    count_episodes += 1
                    if count_episodes == num_episodes:
                        break
            
            # check evaluating done
            if count_episodes == num_episodes:
                break
        
        # return results
        return eval_info
    
    def run(
        self,         
        epochs: int = 1000,
        num_cycles: int = 100,
        num_eval_episodes: int = 20,
        r_mix: float = 0.5,
        *,
        # train params
        num_ppo_epochs: int = 4,
        batch_size: int = 256,
        discounted_factor: float = 0.98,
        gae_lambda: float = 0.95,
        # experiment ckpt
        project_path: Union[str, Path]='',
        name: str='exp',
        resume_path: Union[str, Path] = '',
        save_every_best: bool = False,
        reward_func: Optional[Callable] = None
    ):
        num_envs = self.envs.num_envs
        assert (num_cycles % num_envs) == 0, ValueError(f'Invalid num_cycles ({num_cycles}). It must be divided by num_envs ({num_envs}).')
        t0 = time.time()

        # make experiement directory --------------------------------------------
        exp_dir = self.make_exp_dir(project_path, name, resume_path, subfols=['ckpt', 'policy', 'events'])
        if resume_path != '':
            LOGGER.info(10 * '>' + 10 * '-' + ' LOADING CHECKPOINT ' + 10 * '-' + 10 * '<')
            self.load_ckpt(f=exp_dir / 'ckpt', running_param_names=['start_epoch', 'best_eval', 'total_updates', 'time_delta'])
        LOGGER.info(f'*** Experiment directory: {exp_dir.as_posix()}')

        LOGGER.info(10 * '>' + 10 * '-' + ' START TRAINING ' + 10 * '-' + 10 * '<')
        LOGGER.info(f'- Start epoch: {self._start_epoch}')
        LOGGER.info(f'- Best evaluation: {self._best_eval}')
        LOGGER.info(f'- Total updates: {self._total_updates}')
        LOGGER.info(f'- Start time: {format_time(self._time_delta)}')

        with SummaryWriter(log_dir=exp_dir / 'events') as writer:
            for epoch in range(self._start_epoch, epochs + 1):
                # run explorating and training --------------------------------------------
                epoch_t0, train_run_info = time.time(), MeanMetrics()
                for i, cycle in enumerate(range(num_envs, num_cycles + num_envs, num_envs)):
                    cycle_t0 = time.time()
                    # collect on-policy rollouts
                    rollouts, run_info = self.collect_rollouts(r_mix=r_mix, reward_func=reward_func)
                    train_run_info.update(run_info)
                    
                    # train on collected rollouts
                    train_evals = self.train(
                        rollouts=rollouts,
                        num_epochs=num_ppo_epochs,
                        batch_size=batch_size,
                        discounted_factor=discounted_factor,
                        gae_lambda=gae_lambda
                    )
                    self._total_updates += num_ppo_epochs
                    
                    for name, value in train_evals.items():
                        writer.add_scalar(
                            tag=f'train/{name}', scalar_value=value, 
                            global_step=i + (num_cycles // num_envs) * (epoch - 1)
                        )
                    # show results
                    LOGGER.info(format_tabulate(
                        title='RUNNING RESULTS',
                        subtitle=f'epoch: {epoch}/{epochs}, cycle: {cycle}/{num_cycles}, run_time: {round(time.time() - cycle_t0, 2)}s',
                        results=run_info,
                        tail_info=f'total_updates: {self._total_updates}, duration: {format_time(time.time() - t0 + self._time_delta)}'
                    ) + '\n')

                # show train result per epoch
                LOGGER.info(format_tabulate(
                    title=f'TRAINING RESULTS',
                    subtitle=f'epoch: {epoch}/{epochs}, run_time: {format_time(time.time() - epoch_t0)}',
                    results=train_run_info,
                    tail_info=f'total_updates: {self._total_updates}, duration: {format_time(time.time() - t0 + self._time_delta)}'
                ) + '\n')

                # run evaluating and compute evaluation metric --------------------------------------
                eval_t0 = time.time()
                eval_run_info = self.evaluate(num_eval_episodes, reward_func=reward_func)
                eval_value = eval_run_info['eval_value']
                # show evaluating results per epoch
                LOGGER.info(format_tabulate(
                    title='EVALUATING RESULTS',
                    subtitle=f'epoch: {epoch}/{epochs}, eval: {round(eval_value, 5)}, run_time: {format_time(time.time() - eval_t0)}',
                    results=eval_run_info,
                    tail_info=f'total_updates: {self._total_updates}, duration: {format_time(time.time() - t0 + self._time_delta)}'
                ))
                
                # update training parameters and save checkpoint
                if eval_value >= self._best_eval:
                    self._best_eval = float(eval_value)
                    # Save to experiment directory
                    self.save_policy(
                        f=exp_dir / 'policy', 
                        best_eval=self._best_eval, 
                        global_step=epoch, 
                        save_every_best=save_every_best
                    )
                    best_policy_dir = Path(project_path) / 'runs' / 'best_policy' / 'lift'
                    best_policy_dir.mkdir(parents=True, exist_ok=True)
                    self.save_policy(
                        f=best_policy_dir,
                        best_eval=self._best_eval,
                        global_step=epoch,
                        save_every_best=False
                    )
                        
                self.save_ckpt(f=exp_dir / 'ckpt', running_params={
                    'start_epoch': epoch + 1,
                    'best_eval': self._best_eval,
                    'total_updates': self._total_updates,
                    'time_delta': time.time() - t0 + self._time_delta,
                })
                for name in eval_run_info.keys():
                    writer.add_scalars(
                        main_tag=f'evaluations/{name}',
                        tag_scalar_dict={'train': train_run_info[name], 'eval': eval_run_info[name]},
                        global_step=epoch
                    )
                LOGGER.info('')
