import math
import torch
import numpy as np
from .common import *

# Isaac Lab imports
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm


###
##### EVENT PART
###

# List of available objects
OBJECT_NAMES = ['cube', 'mustard_bottle', 'drill']

def reset_object_poses(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    """Reset object poses with random object selection per environment."""
    
    # Initialize active_objects buffer if it doesn't exist
    if not hasattr(env, 'active_objects'):
        env.active_objects = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    # Randomly select an object for each resetting environment
    num_resets = len(env_ids)
    selected_objects = torch.randint(0, len(OBJECT_NAMES), (num_resets,), device=env.device)
    env.active_objects[env_ids] = selected_objects
    
    # For each object type, determine which environments should have it active
    for obj_idx, obj_name in enumerate(OBJECT_NAMES):
        # Find environments that should have this object
        mask = (selected_objects == obj_idx)
        active_env_ids = env_ids[mask]
        
        if len(active_env_ids) > 0:
            # Sample poses for active environments
            object_pose = sample_root_state_uniform(
                env=env, env_ids=active_env_ids, 
                pose_range={'x': (-0.2, 0.2), 'y': (-0.2, 0.2), 'yaw': (0, 2 * math.pi)},
                asset_cfg=SceneEntityCfg(obj_name)
            )[0]
            # Make object visible and enable physics
            env.scene[obj_name].write_root_pose_to_sim(object_pose, env_ids=active_env_ids)
            env.scene[obj_name].set_visibility(True, env_ids=active_env_ids)
        
        # Hide inactive objects by making them invisible and moving underground
        inactive_mask = (selected_objects != obj_idx)
        inactive_env_ids = env_ids[inactive_mask]
        
        if len(inactive_env_ids) > 0:
            # Make invisible
            env.scene[obj_name].set_visibility(False, env_ids=inactive_env_ids)
            # Move underground to avoid any physics interactions
            current_state = env.scene[obj_name].data.root_state_w[inactive_env_ids].clone()
            current_state[:, 2] = -10.0  # Move 10 meters underground
            env.scene[obj_name].write_root_pose_to_sim(current_state[:, :7], env_ids=inactive_env_ids)


def clip_object_ranges(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor
):
    """Clip object positions to workspace bounds for all active objects."""
    # Handle None case
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    # Clip each object type for the environments that have it active
    if hasattr(env, 'active_objects'):
        for obj_idx, obj_name in enumerate(OBJECT_NAMES):
            # Find environments that have this object active
            mask = (env.active_objects[env_ids] == obj_idx)
            active_env_ids = env_ids[mask]
            
            if len(active_env_ids) > 0:
                clip_object_xy_range(
                    env=env, env_ids=active_env_ids,
                    x_range=(0.05, 0.55), y_range=(-0.3, 0.3), z_thresh=0.05,
                    asset_cfg=SceneEntityCfg(obj_name)
                )
    

@configclass
class EventsCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=reset_scene_to_default, mode="reset")

    reset_objects = EventTerm(
        func=reset_object_poses,
        mode='reset'
    )

    clip_object_positions = EventTerm(
        func=clip_object_ranges,
        mode='interact'
    )