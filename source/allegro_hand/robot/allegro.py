# Copyright (c) 2024-2025, AllegroCrx Project
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the AllegroCrx robot (CRX-10iAL + Allegro Hand).

The following configurations are available:

* :obj:`ALLEGRO_CRX_CFG`: AllegroCrx robot configuration with CRX-10iAL
  arm and Allegro Hand.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to the USD file
ASSETS_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

##
# Configuration
##


ALLEGRO_CRX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_DATA_DIR}/Robots/Allegro.usd",
        activate_contact_sensors=True,  # Enable for hand grasping detection
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Initial orientation (identity quaternion - no rotation)
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        # CRX-10iAL arm joints - all at 0 degrees
        joint_pos={
            # CRX-10iAL Arm (6-DOF)
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
            # Allegro Hand - Index finger
            "index_joint_0": 0.0,
            "index_joint_1": 0.0,
            "index_joint_2": 0.0,
            "index_joint_3": 0.0,
            # Allegro Hand - Middle finger
            "middle_joint_0": 0.0,
            "middle_joint_1": 0.0,
            "middle_joint_2": 0.0,
            "middle_joint_3": 0.0,
            # Allegro Hand - Ring finger
            "ring_joint_0": 0.0,
            "ring_joint_1": 0.0,
            "ring_joint_2": 0.0,
            "ring_joint_3": 0.0,
            # Allegro Hand - Thumb
            "thumb_joint_0": 0.0,
            "thumb_joint_1": 0.0,
            "thumb_joint_2": 0.0,
            "thumb_joint_3": 0.0,
        },
        # Set initial joint velocities to zero
        joint_vel={".*": 0.0},
    ),
    actuators={
        # CRX-10iAL Arm actuators
        # CRX-10iAL is a collaborative robot with payload capacity of 10kg
        # Joint torque limits and inertia decrease from base to end-effector
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            effort_limit={
                "joint_1": 100.0,  # Base rotation - highest torque
                "joint_2": 100.0,  # Shoulder - high torque
                "joint_3": 50.0,   # Elbow - medium torque
                "joint_4": 25.0,   # Wrist 1 - lower torque
                "joint_5": 25.0,   # Wrist 2 - lower torque
                "joint_6": 25.0,   # Wrist 3 - lowest torque
            },
            velocity_limit={
                "joint_1": 3.14,  # ~180 deg/s
                "joint_2": 3.14,
                "joint_3": 3.14,
                "joint_4": 3.14,
                "joint_5": 3.14,
                "joint_6": 3.14,
            },
            stiffness={
                "joint_1": 400.0,  # High stiffness for base
                "joint_2": 400.0,  # High stiffness for shoulder
                "joint_3": 300.0,  # Medium stiffness for elbow
                "joint_4": 200.0,  # Lower stiffness for wrist
                "joint_5": 200.0,
                "joint_6": 200.0,
            },
            damping={
                "joint_1": 80.0,
                "joint_2": 80.0,
                "joint_3": 60.0,
                "joint_4": 40.0,
                "joint_5": 40.0,
                "joint_6": 40.0,
            },
        ),
        # Allegro Hand actuators
        # Allegro Hand has 16 DOF with 4 fingers (3 fingers + 1 thumb)
        # Each finger has 4 joints with small, precise movements
        "hand": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint_.*"],
            # Small torque for hand joints (typical for Allegro)
            effort_limit=0.7,
            velocity_limit=2.0,  # Fast finger movements
            stiffness={
                # Index finger
                "index_joint_0": 40.0,  # Base joint - slightly higher
                "index_joint_1": 35.0,
                "index_joint_2": 35.0,
                "index_joint_3": 30.0,  # Tip joint - slightly lower
                # Middle finger
                "middle_joint_0": 40.0,
                "middle_joint_1": 35.0,
                "middle_joint_2": 35.0,
                "middle_joint_3": 30.0,
                # Ring finger
                "ring_joint_0": 40.0,
                "ring_joint_1": 35.0,
                "ring_joint_2": 35.0,
                "ring_joint_3": 30.0,
                # Thumb (may need different values for opposition movement)
                "thumb_joint_0": 40.0,
                "thumb_joint_1": 35.0,
                "thumb_joint_2": 35.0,
                "thumb_joint_3": 30.0,
            },
            damping={
                # Index finger
                "index_joint_0": 10.0,
                "index_joint_1": 10.0,
                "index_joint_2": 10.0,
                "index_joint_3": 10.0,
                # Middle finger
                "middle_joint_0": 10.0,
                "middle_joint_1": 10.0,
                "middle_joint_2": 10.0,
                "middle_joint_3": 10.0,
                # Ring finger
                "ring_joint_0": 10.0,
                "ring_joint_1": 10.0,
                "ring_joint_2": 10.0,
                "ring_joint_3": 10.0,
                # Thumb
                "thumb_joint_0": 10.0,
                "thumb_joint_1": 10.0,
                "thumb_joint_2": 10.0,
                "thumb_joint_3": 10.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of AllegroCrx robot (CRX-10iAL arm + Allegro Hand)."""
