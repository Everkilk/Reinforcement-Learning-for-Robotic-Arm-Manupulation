from typing import Optional, Tuple
import torch

import isaaclab.utils.math as math_utils

###
##### REWARD FUNCTION
###

class FrankaCudeLiftReward:
    """Custom reward function for Franka-Shadow cube lifting task."""
    
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
        Compute rewards for the lift task.
        
        Args:
            next_observations: Current and next observations
            encoded_goals: Target encoded goals
            stage_id: Current stage (0=reach, 1=lift)
            metas: Meta information for penalties
        """
        # Pre-process
        if isinstance(next_observations, (list, tuple)):
            next_observations = next_observations[0][:, -1]
            
        # Compute penalty rewards if objects are in invalid ranges
        # penalty_rewards = (-1.0 - (metas.sum(dim=-1) / metas.size(-1))) if metas is not None else -1.0
        penalty_rewards = -1.0
    
        # Extract observation components
        object_pos, object_orient = next_observations[:, :3], next_observations[:, 3:6]
        object_quat = math_utils.quat_from_euler_xyz(object_orient[:, 0], object_orient[:, 1], object_orient[:, 2])
        hand_ee_pos, hand_ee_quat = next_observations[:, 6:9], math_utils.quat_from_euler_xyz(*next_observations[:, 9:12].T)
        finger_positions = next_observations[:, 12:27].reshape(-1, 5, 3)
                
        # Compute extrinsic rewards based on stage
        if stage_id == 0:  # Reach object
            device = hand_ee_pos.device
            
            hand_to_obj = object_pos - hand_ee_pos
            hand_obj_dist = torch.norm(hand_to_obj, dim=-1)
            finger_obj_dists = torch.norm(object_pos.unsqueeze(1) - finger_positions, dim=-1).sum(dim=-1)

            # Reach reward: Make sure hand gets close to object
            r_reach = -1.0 * hand_obj_dist - 0.5 * finger_obj_dists
            r_proximity = torch.where(
                hand_obj_dist < 0.12,
                torch.tensor(0.5, device=device),
                torch.tensor(0.0, device=device)
            )
            
            # Reach reward: Make sure palm facing object
            hand_to_obj_normalized = hand_to_obj / (hand_obj_dist.unsqueeze(-1) + 1e-6)
            palm_normal = math_utils.quat_apply(
                hand_ee_quat,
                torch.tensor([[0., 0., -1.]], device=device).expand(hand_ee_quat.shape[0], -1)
            )
            palm_alignment = (palm_normal * hand_to_obj_normalized).sum(dim=-1)
            r_palm_facing = torch.where(
                palm_alignment > 0.3,  
                0.5 * palm_alignment,
                torch.tensor(0.0, device=device)
            )
            
            # Terminal condition: distance + palm orientation
            terminals = ((hand_obj_dist < 0.05) & (palm_alignment > 0.3)).float()
            task_rewards = r_reach + r_proximity + r_palm_facing + terminals
        elif stage_id == 1:  # Lift to goal
            distances = (object_pos - encoded_goals[:, 3:6]).norm(dim=-1)
            terminals = (distances <= 0.03).float()
            task_rewards = terminals
        else:
            raise ValueError(f"Invalid stage_id: {stage_id}")
        
        # Compute final rewards
        rewards = self.scale_factor * (task_rewards + penalty_rewards)
        
        return rewards, terminals