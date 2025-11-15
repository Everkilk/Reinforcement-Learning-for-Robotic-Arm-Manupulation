from typing import Optional, Tuple
import torch

import isaaclab.utils.math as math_utils

###
##### DENSE REWARD FUNCTION FOR PPO
###

class FrankaCudeLiftRewardDense:
    """Dense reward function for Franka-Shadow cube lifting task - optimized for PPO."""
    
    def __init__(self, scale_factor: float = 1.0):
        self.scale_factor = scale_factor
        self.object_lengths = torch.tensor([0.06, 0.06, 0.06])
        self.grasp_range = torch.tensor([0.12, 0.12, 0.09])
    
    def __call__(
        self, 
        next_observations: torch.Tensor, 
        encoded_goals: torch.Tensor, 
        stage_id: int, 
        metas: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DENSE rewards for the lift task (PPO-friendly).
        
        Args:
            next_observations: Current and next observations
            encoded_goals: Target encoded goals
            stage_id: Current stage (0=reach, 1=lift)
            metas: Meta information for penalties
        """
        # Pre-process
        if isinstance(next_observations, (list, tuple)):
            next_observations = next_observations[0][:, -1]
            
        penalty_rewards = -1.0
    
        # Extract observation components
        object_pos, object_orient = next_observations[:, :3], next_observations[:, 3:6]
        object_quat = math_utils.quat_from_euler_xyz(object_orient[:, 0], object_orient[:, 1], object_orient[:, 2])
        hand_ee_pos, hand_ee_quat = next_observations[:, 6:9], math_utils.quat_from_euler_xyz(*next_observations[:, 9:12].T)
        finger_positions = next_observations[:, 12:27].reshape(-1, 5, 3)
                
        # Compute DENSE extrinsic rewards based on stage
        if stage_id == 0:  # Reach object - DENSE VERSION
            grasp_range = self.grasp_range.to(encoded_goals.device)
            object_lengths = self.object_lengths.to(encoded_goals.device)
            
            # Dense distance reward - hand to goal
            hand_to_goal = (hand_ee_pos - encoded_goals[:, :3]).norm(dim=-1)
            distance_reward = 1.0 - torch.tanh(hand_to_goal * 5.0)  # Dense: 1.0 when close, ~0 when far
            
            # Finger spreading reward (prevent sticky fingers)
            finger_distances = (
                (finger_positions.unsqueeze(1) - finger_positions.unsqueeze(2)).norm(dim=-1, p=2)
            ).permute(1, 2, 0)[torch.triu_indices(5, 5, offset=1).tolist()].permute(1, 0)
            not_sticky_scores = ((finger_distances[:, :4] / 0.025).clamp(0.0, 1.0).square() - 1.0).mean(dim=-1) \
                                + ((finger_distances[:, 4:] / 0.01).clamp(0.0, 1.0).square() - 1.0).mean(dim=-1)
            
            # Finger-to-object proximity reward (dense)
            delta_obj2fingers = torch.stack([
                math_utils.subtract_frame_transforms(q01=object_quat, t01=object_pos, t02=finger_pos)[0]
                for finger_pos in finger_positions.permute(1, 0, 2)
            ], dim=1).clamp(-grasp_range, grasp_range) / object_lengths
            reach_scores = 1.0 - ((delta_obj2fingers.norm(p=25, dim=-1).clamp(0.5, None) - 0.5) / ((grasp_range / object_lengths).norm(p=25) - 0.5)).sqrt().mean(dim=-1)
            
            # Combine all dense components
            combined_scores = 0.3 * distance_reward + 0.5 * reach_scores + 0.2 * (not_sticky_scores + 1.0)
            
            # Terminal bonus
            delta_goal = math_utils.subtract_frame_transforms(q01=hand_ee_quat, t01=hand_ee_pos, t02=encoded_goals[:, :3])[0]
            terminals = (delta_goal.abs() <= object_lengths / 2).all(dim=-1).float()
            
            task_rewards = combined_scores + 2.0 * terminals  # Big bonus for reaching goal
            
        elif stage_id == 1:  # Lift to goal - DENSE VERSION
            # Dense distance-based reward (instead of binary)
            distances = (object_pos - encoded_goals[:, 3:6]).norm(dim=-1)
            distance_reward = 1.0 - torch.tanh(distances * 20.0)  # Dense: 1.0 when very close, ~0 when far
            
            # Height reward - encourage lifting
            height_diff = object_pos[:, 2] - encoded_goals[:, 3:6][:, 2]
            height_reward = torch.tanh(height_diff.clamp(min=0) * 10.0)  # Reward for being above goal
            
            # Terminal bonus
            terminals = (distances <= 0.03).float()
            
            task_rewards = 0.6 * distance_reward + 0.2 * height_reward + 2.0 * terminals
            
        else:
            raise ValueError(f"Invalid stage_id: {stage_id}")
        
        # Compute final rewards
        rewards = self.scale_factor * (task_rewards + penalty_rewards)
        
        return rewards, terminals
