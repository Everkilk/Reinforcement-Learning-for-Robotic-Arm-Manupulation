"""
ResDex-style Reward Function for Dexterous Grasping.

Based on: "Efficient Residual Learning with Mixture-of-Experts for Universal Dexterous Grasping"
(Huang et al., 2024)

This reward function implements the multi-stage reward structure from the ResDex paper:
- Base Policy: r = r^pose + r^task (r^task = r^reach + r^lift + r^move + r^bonus)
- Hyper Policy Stage 1: r = r^task + r^proposal
- Hyper Policy Stage 2: r = r^lift + r^move + r^bonus (loosened)
"""

from typing import Optional, Tuple
import torch
import isaaclab.utils.math as math_utils


###
##### RESDEX REWARD FUNCTION
###

class ResDexReward:
    """
    ResDex-style reward function for dexterous grasping.
    
    Implements rewards for both Base Policy and Hyper Policy training as described in the paper.
    All stage-specific logic is organized within each stage block for clarity.
    """
    
    def __init__(
        self,
        # General parameters
        scale_factor: float = 1.0,
        training_mode: str = 'base',  # 'base', 'hyper_stage1', 'hyper_stage2'
        
        # Pose reward parameters (Base Policy)
        pose_weight: float = 0.05,
        
        # Reach reward parameters
        hand_reach_weight: float = 1.0,
        finger_reach_weight: float = 0.5,
        
        # Lift reward parameters
        lift_base_reward: float = 0.1,
        lift_force_scale: float = 0.1,
        finger_dist_threshold: float = 0.6,
        hand_dist_threshold: float = 0.12,
        
        # Move reward parameters
        move_base_reward: float = 0.9,
        move_dist_scale: float = 2.0,
        joint_diff_threshold: float = 6.0,
        
        # Bonus reward parameters
        bonus_dist_threshold: float = 0.05,
        bonus_scale: float = 10.0,
        
        # Object parameters
        object_lengths: Tuple[float, float, float] = (0.06, 0.06, 0.06),
        grasp_range: Tuple[float, float, float] = (0.12, 0.12, 0.09),
    ):
        """
        Initialize ResDex reward function.
        
        Args:
            scale_factor: Overall reward scaling
            training_mode: 'base' for base policy, 'hyper_stage1' for hyper policy stage 1,
                          'hyper_stage2' for hyper policy stage 2 (loosened)
            pose_weight: Weight for joint position penalty (base policy)
            hand_reach_weight: Weight for hand-to-object distance
            finger_reach_weight: Weight for finger-to-object distance
            lift_base_reward: Base reward for lifting
            lift_force_scale: Scale for lift force reward
            finger_dist_threshold: Threshold for finger proximity (default 0.6)
            hand_dist_threshold: Threshold for hand proximity (default 0.12)
            move_base_reward: Base reward for moving to target
            move_dist_scale: Scale for move distance penalty
            joint_diff_threshold: Threshold for joint difference (default 6.0)
            bonus_dist_threshold: Distance threshold for bonus (default 0.05)
            bonus_scale: Scale factor for bonus reward
            object_lengths: Object dimensions (x, y, z)
            grasp_range: Valid grasping range
        """
        self.scale_factor = scale_factor
        self.training_mode = training_mode
        
        # Base policy parameters
        self.pose_weight = pose_weight
        
        # Reach parameters
        self.hand_reach_weight = hand_reach_weight
        self.finger_reach_weight = finger_reach_weight
        
        # Lift parameters
        self.lift_base_reward = lift_base_reward
        self.lift_force_scale = lift_force_scale
        self.finger_dist_threshold = finger_dist_threshold
        self.hand_dist_threshold = hand_dist_threshold
        
        # Move parameters
        self.move_base_reward = move_base_reward
        self.move_dist_scale = move_dist_scale
        self.joint_diff_threshold = joint_diff_threshold
        
        # Bonus parameters
        self.bonus_dist_threshold = bonus_dist_threshold
        self.bonus_scale = bonus_scale
        
        # Object parameters
        self.object_lengths = torch.tensor(object_lengths)
        self.grasp_range = torch.tensor(grasp_range)
    
    def __call__(
        self,
        next_observations: torch.Tensor,
        encoded_goals: torch.Tensor,
        stage_id: int,
        metas: Optional[torch.Tensor] = None,
        current_joints: Optional[torch.Tensor] = None,
        target_joints: Optional[torch.Tensor] = None,
        lift_force_z: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute total reward following ResDex paper structure.
        
        Args:
            next_observations: Observations tensor containing:
                - [:, :3] object position
                - [:, 3:6] object orientation (euler)
                - [:, 6:9] hand position
                - [:, 9:12] hand orientation (euler)
                - [:, 12:27] finger positions (5 fingers * 3)
                - [:, 27:] joint positions (optional)
            encoded_goals: Target goals tensor
                - [:, :3] reach target (hand position target, for stage 0)
                - [:, 3:6] lift target (object target position, for stage 1)
            stage_id: Current training stage (0=reach, 1=lift)
            metas: Optional metadata for additional penalties
            current_joints: Current joint positions [B, N] (for base policy)
            target_joints: Target joint positions [B, N] (for base policy)
            lift_force_z: Z-axis lift force [B] (optional)
            
        Returns:
            Tuple of (rewards, terminals)
        """
        # Pre-process observations
        if isinstance(next_observations, (list, tuple)):
            next_observations = next_observations[0][:, -1]
        
        device = next_observations.device
        self.object_lengths = self.object_lengths.to(device)
        self.grasp_range = self.grasp_range.to(device)
        
        # Extract observation components
        object_pos = next_observations[:, :3]
        object_orient = next_observations[:, 3:6]
        object_quat = math_utils.quat_from_euler_xyz(
            object_orient[:, 0], object_orient[:, 1], object_orient[:, 2]
        )
        hand_pos = next_observations[:, 6:9]
        hand_orient = next_observations[:, 9:12]
        hand_quat = math_utils.quat_from_euler_xyz(*hand_orient.T)
        finger_positions = next_observations[:, 12:27].reshape(-1, 5, 3)
        
        # Extract joint positions if available
        if current_joints is None and next_observations.size(-1) > 27:
            current_joints = next_observations[:, 27:]
        
        # Default lift force if not provided
        if lift_force_z is None:
            lift_force_z = torch.zeros(object_pos.size(0), device=device)
        
        # Penalty rewards
        penalty_rewards = -1.0
        
        # Compute rewards based on stage
        if stage_id == 0:  # STAGE 0: REACH OBJECT
            """
            Stage 0 implements reaching behavior to get hand close to object.
            
            For Base Policy:
                r = r^pose + r^reach
                r^pose = -0.05 * ||q - X_joint||_1
                r^reach = -1.0 * ||X_obj - X_hand||_2 - 0.5 * sum(||X_obj - X_finger||_2)
            
            For Hyper Policy:
                r = r^reach (no pose penalty)
            """
            # ===== REACH REWARD (r^reach) =====
            # Hand to object distance
            hand_obj_dist = torch.norm(object_pos - hand_pos, dim=-1, p=2)
            
            # Fingers to object distance (sum over all fingers)
            finger_obj_dists = torch.norm(
                object_pos.unsqueeze(1) - finger_positions, dim=-1, p=2
            ).sum(dim=-1)
            
            r_reach = (
                -self.hand_reach_weight * hand_obj_dist 
                - self.finger_reach_weight * finger_obj_dists
            )
            
            # ===== POSE REWARD (r^pose) - Base Policy Only =====
            r_pose = torch.zeros_like(hand_obj_dist)
            if self.training_mode == 'base' and target_joints is not None and current_joints is not None:
                joint_diff = torch.norm(target_joints - current_joints, dim=-1, p=1)
                r_pose = -self.pose_weight * joint_diff
            
            # ===== PROXIMITY BONUS =====
            # Additional reward when hand is very close
            r_proximity = torch.where(
                hand_obj_dist < self.hand_dist_threshold,
                torch.tensor(0.5, device=device),
                torch.tensor(0.0, device=device)
            )
            
            # ===== PALM ORIENTATION REWARD =====
            # Encourage palm to face the object
            hand_to_obj = object_pos - hand_pos
            hand_to_obj_normalized = hand_to_obj / (hand_obj_dist.unsqueeze(-1) + 1e-6)
            palm_normal = math_utils.quat_apply(
                hand_quat,
                torch.tensor([[0., 0., -1.]], device=device).expand(hand_quat.shape[0], -1)
            )
            palm_alignment = (palm_normal * hand_to_obj_normalized).sum(dim=-1)
            r_palm_facing = torch.where(
                palm_alignment > 0.3,
                0.5 * palm_alignment,
                torch.tensor(0.0, device=device)
            )
            
            # ===== TERMINAL CONDITION =====
            # Success when hand is close to object with good orientation
            terminals = ((hand_obj_dist < 0.05) & (palm_alignment > 0.3)).float()
            
            # ===== TOTAL STAGE 0 REWARD =====
            task_rewards = r_reach + r_pose + r_proximity + r_palm_facing + terminals
            
        elif stage_id == 1:  # STAGE 1: LIFT AND MOVE TO TARGET
            """
            Stage 1 implements lifting and moving object to target.
            
            For Base Policy:
                r = r^pose + r^task
                r^task = r^reach + r^lift + r^move + r^bonus
            
            For Hyper Policy Stage 1:
                r = r^task + r^proposal
                r^task = r^reach + r^lift + r^move + r^bonus
            
            For Hyper Policy Stage 2 (loosened):
                r = r^lift + r^move + r^bonus
            """
            target_pos = encoded_goals[:, 3:6]
            
            # ===== REACH REWARD (r^reach) =====
            # Continue encouraging hand to stay close to object
            hand_obj_dist = torch.norm(object_pos - hand_pos, dim=-1, p=2)
            finger_obj_dists = torch.norm(
                object_pos.unsqueeze(1) - finger_positions, dim=-1, p=2
            ).sum(dim=-1)
            
            r_reach = (
                -self.hand_reach_weight * hand_obj_dist 
                - self.finger_reach_weight * finger_obj_dists
            )
            
            # ===== LIFT REWARD (r^lift) =====
            # f_1 = 1(sum||X_obj - X_finger||_2 <= 0.6) + 1(||X_obj - X_hand||_2 <= 0.12)
            # r^lift = {0.1 + 0.1 * a_z if f_1 = 2, else 0}
            cond_finger = (finger_obj_dists <= self.finger_dist_threshold).float()
            cond_hand = (hand_obj_dist <= self.hand_dist_threshold).float()
            f1 = cond_finger + cond_hand
            
            r_lift = torch.where(
                f1 >= 2.0,
                self.lift_base_reward + self.lift_force_scale * lift_force_z,
                torch.zeros_like(f1)
            )
            
            # ===== MOVE REWARD (r^move) =====
            # Stage 1 (full): f_2 = f_1 + 1(||q - X_joint||_1 <= 6)
            #                 r^move = {0.9 - 2||X_obj - X_target||_2 if f_2 = 3, else 0}
            # Stage 2 (loosened): r^move = {0.9 - 2||X_obj - X_target||_2 if f_1 = 2, else 0}
            obj_target_dist = torch.norm(object_pos - target_pos, dim=-1, p=2)
            
            if self.training_mode == 'hyper_stage2':
                # Loosened condition: only need f1 = 2
                condition = (f1 >= 2.0)
            else:
                # Full condition for base policy or hyper stage 1
                if target_joints is not None and current_joints is not None:
                    joint_diff = torch.norm(target_joints - current_joints, dim=-1, p=1)
                    cond_joint = (joint_diff <= self.joint_diff_threshold).float()
                    f2 = f1 + cond_joint
                    condition = (f2 >= 3.0)
                else:
                    # Fallback to loosened condition if joints not available
                    condition = (f1 >= 2.0)
            
            r_move = torch.where(
                condition,
                self.move_base_reward - self.move_dist_scale * obj_target_dist,
                torch.zeros_like(obj_target_dist)
            )
            
            # ===== BONUS REWARD (r^bonus) =====
            # d_obj = ||X_obj - X_target||_2
            # r^bonus = {1 / (1 + 10 * d_obj) if d_obj <= 0.05, else 0}
            r_bonus = torch.where(
                obj_target_dist <= self.bonus_dist_threshold,
                1.0 / (1.0 + self.bonus_scale * obj_target_dist),
                torch.zeros_like(obj_target_dist)
            )
            
            # ===== POSE REWARD (r^pose) - Base Policy Only =====
            r_pose = torch.zeros_like(obj_target_dist)
            if self.training_mode == 'base' and target_joints is not None and current_joints is not None:
                joint_diff = torch.norm(target_joints - current_joints, dim=-1, p=1)
                r_pose = -self.pose_weight * joint_diff
            
            # ===== TERMINAL CONDITION =====
            # Success when object is at target position
            terminals = (obj_target_dist <= 0.03).float()
            
            # ===== TOTAL STAGE 1 REWARD =====
            if self.training_mode == 'hyper_stage2':
                # Hyper Policy Stage 2 (loosened): no reach, no pose
                task_rewards = r_lift + r_move + r_bonus + terminals
            elif self.training_mode == 'base':
                # Base Policy: pose + reach + lift + move + bonus
                task_rewards = r_pose + r_reach + r_lift + r_move + r_bonus + terminals
            else:  # hyper_stage1 or default
                # Hyper Policy Stage 1: reach + lift + move + bonus (no pose)
                task_rewards = r_reach + r_lift + r_move + r_bonus + terminals
                
        else:
            raise ValueError(f"Invalid stage_id: {stage_id}")
        
        # ===== FINAL REWARDS =====
        rewards = self.scale_factor * (task_rewards + penalty_rewards)
        
        return rewards, terminals
