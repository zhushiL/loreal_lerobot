#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for Dobot Nova5 robot."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

from lerobot.robots.dobot_nova5.TCP_IP_Python_V4.dobot_api import DobotApiFeedBack,DobotApiDashboard

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.dobot_nova5.config_xense_gripper import GripperConfig, SensorOutputType


class ControlMode(str, Enum):
    """Control mode for Dobot Nova5.

    JOINT_MOTION:
        Joint motion control.
        - Action: joint positions (7D) + gripper (1D) = 8D
        - Observation: joint positions (7D) + gripper (1D) = 8D

    CARTESIAN_MOTION:
        Cartesian motion control.
        - Action: TCP pose (9D) + gripper (1D) = 10D
        - Observation: TCP pose (9D) + gripper (1D) = 10D
    """

    JOINT_MOTION = "joint_motion_control"
    CARTESIAN_MOTION = "cartesian_motio``n_control"

@RobotConfig.register_subclass("dobot_nova5")
@dataclass
class DobotNova5Config(RobotConfig):
    """Configuration for Dobot Nova5 robot.

    The Dobot Nova5 is a 6-DOF collaborative robot with force sensing capabilities.

    Attributes:
        robot_ip: str = "192.168.1.1"  # Robot IP address
        control_mode: ControlMode = ControlMode.CARTESIAN_MOTION
        control_frequency: float = 100.0  # Hz
        cameras: dict[str, CameraConfig] = field(default_factory=dict)

        # Joint motion constraints (for joint motion control mode)
        aheadtime: float = 50.0  # PID controller D (20.0-100.0, default 50.0)
        gain: float = 500.0  # PID controller P (200.0-1000.0, default 500.0)

        # Cartesian motion parameters
        aheadtime: float = 50.0  # PID controller D (20.0-100.0, default 50.0)
        gain: float = 500.0  # PID controller P (200.0-1000.0, default 500.0)
    """

    # Robot identification
    robot_ip: str = "192.168.1.1"  # Robot IP address
    dashboardPort: int = 29999
    feedPortFour: int = 30004

    # Control settings
    # control_mode: JOINT_MOTION or CARTESIAN_MOTION
    control_mode: ControlMode = ControlMode.CARTESIAN_MOTION

    # NRT mode supports 1-100 Hz
    control_frequency: float = 100.0  # Hz

    # Connection behavior
    go_to_start: bool = (
        True  # If True, move robot to start position after connecting. If False, stay at current position.
    )

    aheadtime: float = 50.0  # PID controller D (20.0-100.0, default 50.0)
    gain: float = 500.0  # PID controller P (200.0-1000.0, default 500.0)

    home_point_list = [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]
    # Start position parameters (for MoveJ primitive)
    # Joint positions in degrees (factory-defined home position)
    start_position_degree: list[float] = field(
        default_factory=lambda: [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0]
    )
    # Joint velocity scale for moving to start position (1-100, default 30)
    start_vel_scale: int = 30


    # ======================== Xense Gripper (end-effector) settings ==========
    # Whether to use the Xense Gripper end-effector
    # If False, xense_gripper will be None and no gripper functionality is available
    use_gripper: bool = True

    # Gripper identification (MAC address / serial number)
    xense_gripper_mac_addr: str = "e2b26adbb104"

    # Tactile sensor settings
    xense_gripper_rectify_size: tuple[int, int] = (400, 700)
    xense_gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    xense_gripper_sensor_keys: dict[str, str] = field(
        default_factory=lambda: {
            "OG000657": "right_tactile",
            "OG000450": "left_tactile",
        }
    )

    # Gripper normalization: raw_pos / gripper_max_pos -> [0, 1]
    gripper_min_pos: float = 0.0
    gripper_max_pos: float = 85.0

    # Gripper control parameters for set_position()
    gripper_velocity: float = 80.0  # Maximum velocity mm/s
    gripper_force: float = 20.0  # Maximum force N

    # Initialize gripper to fully open on connect
    xense_gripper_init_open: bool = True

    # Auto-created in __post_init__ from xense_gripper_* parameters (do not set directly)
    xense_gripper: GripperConfig | None = field(default=None, init=False)


    # ======================== Camera Configuration ========================
    
    # RealSense cameras (2 cameras recommended: main + wrist)
    cameras: Dict[str, RealSenseCameraConfig] = field(default_factory=lambda: {
        # # Main external camera
        # "head": RealSenseCameraConfig(
        #     serial_number_or_name="135522074323",
        #     fps=30,
        #     width=640,
        #     height=480,
        #     color_mode=ColorMode.RGB
        # ),
        # # left_wrist camera
        # "left_wrist": RealSenseCameraConfig(
        #     serial_number_or_name="249322063436",
        #     fps=30,
        #     width=640,
        #     height=480,
        #     color_mode=ColorMode.RGB
        # )
        # # right_wrist camera
        # "right_wrist": RealSenseCameraConfig(
        #     serial_number_or_name="249322063436",
        #     fps=30,
        #     width=640,
        #     height=480,
        #     color_mode=ColorMode.RGB
        # )
    })

    def __post_init__(self):
        super().__post_init__()

        # Validate control frequency (NRT mode: 1-100 Hz)
        if not 1 <= self.control_frequency <= 100:
            raise ValueError(
                f"control_frequency must be between 1 and 100 Hz for NRT mode, got {self.control_frequency}"
            )

        # Validate joint parameters have correct length (7-DOF robot)
        if len(self.joint_max_vel) != 7:
            raise ValueError(f"joint_max_vel must have 7 elements, got {len(self.joint_max_vel)}")
        if len(self.joint_max_acc) != 7:
            raise ValueError(f"joint_max_acc must have 7 elements, got {len(self.joint_max_acc)}")

        # Validate Cartesian/force parameters have correct length (6-DOF)
        if len(self.force_control_axis) != 6:
            raise ValueError(f"force_control_axis must have 6 elements, got {len(self.force_control_axis)}")
        if len(self.max_contact_wrench) != 6:
            raise ValueError(f"max_contact_wrench must have 6 elements, got {len(self.max_contact_wrench)}")
        if len(self.target_wrench) != 6:
            raise ValueError(f"target_wrench must have 6 elements, got {len(self.target_wrench)}")

        # Validate start position parameters
        if len(self.start_position_degree) != 7:
            raise ValueError(
                f"start_position_degree must have 7 elements, got {len(self.start_position_degree)}"
            )
        if not 1 <= self.start_vel_scale <= 100:
            raise ValueError(f"start_vel_scale must be between 1 and 100, got {self.start_vel_scale}")

        # Create XenseGripperConfig from exposed parameters (only if use_gripper=True)
        if self.use_gripper:
            self.xense_gripper = GripperConfig(
                mac_addr=self.xense_gripper_mac_addr,
                rectify_size=self.xense_gripper_rectify_size,
                sensor_output_type=self.xense_gripper_sensor_output_type,
                sensor_keys=self.xense_gripper_sensor_keys,
                gripper_velocity=self.gripper_velocity,
                gripper_force=self.gripper_force,
                gripper_min_pos=self.gripper_min_pos,
                gripper_max_pos=self.gripper_max_pos,
                init_open=self.xense_gripper_init_open,
            )
        else:
            self.xense_gripper = None
