import math
import torch
from .common import *

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm


###
##### EVENT PART
###
def reset_random_object_selection(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    """Randomly select which object (cubic or mustard) to use for each environment.
    The non-selected object will be moved far away (invisible and non-interactable).
    """
    # Initialize object selection tensor if it doesn't exist
    if not hasattr(env, 'active_object_type'):
        env.active_object_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Randomly select object type for each environment: 0 = cubic, 1 = mustard
    env.active_object_type[env_ids] = torch.randint(0, 2, (len(env_ids),), device=env.device)
    
    # Get masks for which environments use which object
    cubic_envs = env_ids[env.active_object_type[env_ids] == 0]
    mustard_envs = env_ids[env.active_object_type[env_ids] == 1]
    
    # Position for hiding objects (far away and below ground)
    hide_pos = torch.tensor([100.0, 100.0, -100.0], device=env.device)
    hide_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
    # Hide mustard bottles for cubic environments
    if len(cubic_envs) > 0:
        # Create pose tensor (7 dims: pos + quat)
        mustard_pose = torch.zeros(len(cubic_envs), 7, device=env.device)
        mustard_pose[:, :3] = hide_pos
        mustard_pose[:, 3:7] = hide_rot
        env.scene['object_mustard'].write_root_pose_to_sim(mustard_pose, env_ids=cubic_envs)
    
    # Hide cubic for mustard environments
    if len(mustard_envs) > 0:
        # Create pose tensor (7 dims: pos + quat)
        cubic_pose = torch.zeros(len(mustard_envs), 7, device=env.device)
        cubic_pose[:, :3] = hide_pos
        cubic_pose[:, 3:7] = hide_rot
        env.scene['object_cubic'].write_root_pose_to_sim(cubic_pose, env_ids=mustard_envs)


def reset_object_poses(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    """Reset object poses based on which object is active for each environment."""
    # Ensure object selection exists
    if not hasattr(env, 'active_object_type'):
        env.active_object_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Get masks for which environments use which object
    cubic_envs = env_ids[env.active_object_type[env_ids] == 0]
    mustard_envs = env_ids[env.active_object_type[env_ids] == 1]
    
    # Reset cubic object poses for cubic environments
    if len(cubic_envs) > 0:
        cubic_pose = sample_root_state_uniform(
            env=env, env_ids=cubic_envs, 
            pose_range={'x': (-0.2, 0.2), 'y': (-0.2, 0.2), 'yaw': (0, 2 * math.pi)},
            asset_cfg=SceneEntityCfg('object_cubic')
        )[0]
        env.scene['object_cubic'].write_root_pose_to_sim(cubic_pose, env_ids=cubic_envs)
    
    # Reset mustard object poses for mustard environments
    if len(mustard_envs) > 0:
        mustard_pose = sample_root_state_uniform(
            env=env, env_ids=mustard_envs, 
            pose_range={'x': (-0.2, 0.2), 'y': (-0.2, 0.2), 'yaw': (0, 2 * math.pi)},
            asset_cfg=SceneEntityCfg('object_mustard')
        )[0]
        env.scene['object_mustard'].write_root_pose_to_sim(mustard_pose, env_ids=mustard_envs)


def clip_object_ranges(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    """Clip object positions for active objects only."""
    # Handle None case
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    # Ensure object selection exists
    if not hasattr(env, 'active_object_type'):
        env.active_object_type = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Get masks for which environments use which object
    cubic_envs = env_ids[env.active_object_type[env_ids] == 0]
    mustard_envs = env_ids[env.active_object_type[env_ids] == 1]
    
    # Clip cubic positions
    if len(cubic_envs) > 0:
        clip_object_xy_range(
            env=env, env_ids=cubic_envs,
            x_range=(0.05, 0.55), y_range=(-0.3, 0.3), z_thresh=0.05,
            asset_cfg=SceneEntityCfg('object_cubic')
        )
    
    # Clip mustard positions
    if len(mustard_envs) > 0:
        clip_object_xy_range(
            env=env, env_ids=mustard_envs,
            x_range=(0.05, 0.55), y_range=(-0.3, 0.3), z_thresh=0.05,
            asset_cfg=SceneEntityCfg('object_mustard')
        )
    

@configclass
class EventsCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")

    # First randomly select which object to use
    select_random_object = EventTerm(
        func=reset_random_object_selection,
        mode='reset'
    )
    
    # Then reset the selected object's pose
    reset_objects = EventTerm(
        func=reset_object_poses,
        mode='reset'
    )

    clip_object_positions = EventTerm(
        func=clip_object_ranges,
        mode='interact'
    )
