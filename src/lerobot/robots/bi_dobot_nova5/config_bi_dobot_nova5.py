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

"""Configuration for BiDobot Nova5 robot."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

from lerobot.robots.bi_dobot_nova5.TCP_IP_Python_V4.dobot_api import DobotApiFeedBack,DobotApiDashboard

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig
from lerobot.robots.bi_dobot_nova5.config_xense_gripper import GripperConfig, SensorOutputType


class ControlMode(str, Enum):
    """Control mode for BiDobot Nova5.

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

@RobotConfig.register_subclass("bi_dobot_nova5")
@dataclass
class BiDobotNova5Config(RobotConfig):
    """Configuration for BiDobot Nova5 robot.

    The BiDobot Nova5 is a 6-DOF collaborative robot with force sensing capabilities.

    Attributes:
        left_robot_ip: str = "192.168.5.101"  # Robot IP address
        left_dashboardPort: int = 29999
        left_feedPortFour: int = 30004
        right_robot_ip: str = "192.168.5.102"  # Robot IP address
        right_dashboardPort: int = 29999
        right_feedPortFour: int = 30004
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
    left_robot_ip: str = "192.168.5.101"  # Robot IP address
    left_dashboardPort: int = 29999
    left_feedPortFour: int = 30004
    right_robot_ip: str = "192.168.5.102"  # Robot IP address
    right_dashboardPort: int = 29999
    right_feedPortFour: int = 30004

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

    left_home_point_list = [-90, 0, -90, 0, 90, 0]
    right_home_point_list = [270, 0, 90, 0, -90, 0]
    # Start position parameters (for MoveJ primitive)
    # Joint positions in degrees (factory-defined home position)
    left_start_position_degree: list[float] = field(default_factory=lambda: [-90, 0, -90, 0, 90, 0])
    right_start_position_degree: list[float] = field(default_factory=lambda: [270, 0, 90, 0, -90, 0])
    # Joint velocity scale for moving to start position (1-100, default 30)
    start_vel_scale: int = 30


    # ======================== Xense Gripper (end-effector) settings ==========
    # Whether to use the Xense Gripper end-effector
    # If False, xense_gripper will be None and no gripper functionality is available
    use_left_gripper: bool = False
    use_right_gripper: bool = False

    # Gripper identification (MAC address / serial number)
    xense_left_gripper_mac_addr: str = "e2b26adbb104"
    xense_right_gripper_mac_addr: str = "e2b26adbb104"

    # Tactile sensor settings
    xense_left_gripper_rectify_size: tuple[int, int] = (400, 700)
    xense_right_gripper_rectify_size: tuple[int, int] = (400, 700)
    xense_left_gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    xense_right_gripper_sensor_output_type: SensorOutputType = SensorOutputType.RECTIFY
    xense_left_gripper_sensor_keys: dict[str, str] = field(
        default_factory=lambda: {
            "OG000657": "right_tactile",
            "OG000450": "left_tactile",
        }
    )
    xense_right_gripper_sensor_keys: dict[str, str] = field(
        default_factory=lambda: {
            "OG000657": "right_tactile",
            "OG000450": "left_tactile",
        }
    )

    # Gripper normalization: raw_pos / gripper_max_pos -> [0, 1]
    left_gripper_min_pos: float = 0.0
    right_gripper_min_pos: float = 0.0
    left_gripper_max_pos: float = 85.0
    right_gripper_max_pos: float = 85.0

    # Gripper control parameters for set_position()
    left_gripper_velocity: float = 80.0  # Maximum velocity mm/s
    right_gripper_velocity: float = 80.0  # Maximum velocity mm/s
    left_gripper_force: float = 20.0  # Maximum force N
    right_gripper_force: float = 20.0  # Maximum force N

    # Initialize gripper to fully open on connect
    left_gripper_init_open: bool = True
    right_gripper_init_open: bool = True

    # Auto-created in __post_init__ from xense_gripper_* parameters (do not set directly)
    xense_left_gripper: GripperConfig | None = field(default=None, init=False)
    xense_right_gripper: GripperConfig | None = field(default=None, init=False)


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

        # Validate start position parameters
        if len(self.left_start_position_degree) != 6:
            raise ValueError(f"left_start_position_degree must have 6 elements, got {len(self.left_start_position_degree)}")
        if len(self.right_start_position_degree) != 6:
            raise ValueError(f"right_start_position_degree must have 6 elements, got {len(self.right_start_position_degree)}")
        if len(self.left_home_point_list) != 6:
            raise ValueError(f"left_home_point_list must have 6 elements, got {len(self.left_home_point_list)}")
        if len(self.right_home_point_list) != 6:
            raise ValueError(f"right_home_point_list must have 6 elements, got {len(self.right_home_point_list)}")
        if not 1 <= self.start_vel_scale <= 100:
            raise ValueError(f"start_vel_scale must be between 1 and 100, got {self.start_vel_scale}")

        # Create XenseGripperConfig from exposed parameters (only if use_gripper=True)
        if self.use_left_gripper:
            self.xense_left_gripper = GripperConfig(
                mac_addr=self.xense_left_gripper_mac_addr,
                rectify_size=self.xense_left_gripper_rectify_size,
                sensor_output_type=self.xense_left_gripper_sensor_output_type,
                sensor_keys=self.xense_left_gripper_sensor_keys,
                gripper_velocity=self.left_gripper_velocity,
                gripper_force=self.left_gripper_force,
                gripper_min_pos=self.left_gripper_min_pos,
                gripper_max_pos=self.left_gripper_max_pos,
                init_open=self.left_gripper_init_open,
            )
        if self.use_right_gripper:
            self.xense_right_gripper = GripperConfig(
                mac_addr=self.xense_right_gripper_mac_addr,
                rectify_size=self.xense_right_gripper_rectify_size,
                sensor_output_type=self.xense_right_gripper_sensor_output_type,
                sensor_keys=self.xense_right_gripper_sensor_keys,
                gripper_velocity=self.right_gripper_velocity,
                gripper_force=self.right_gripper_force,
                gripper_min_pos=self.right_gripper_min_pos,
                gripper_max_pos=self.right_gripper_max_pos,
                init_open=self.right_gripper_init_open,
            )

        if not self.use_left_gripper:
            self.xense_left_gripper = None
        if not self.use_right_gripper:
            self.xense_right_gripper = None