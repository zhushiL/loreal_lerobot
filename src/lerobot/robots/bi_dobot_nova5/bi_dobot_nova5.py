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

"""Dobot Nova5 robot implementation for LeRobot.

This module provides integration with Dobot Nova5 6-DOF collaborative robot,
supporting two control modes (NRT = Non-Real-Time for Python API):

1. JOINT_MOTION:
   - Action: joint positions (7D) + gripper (1D) = 8D
   - Observation: joint positions (7D) + gripper (1D) = 8D

2. CARTESIAN_MOTION:
   - Uses 6D rotation representation (r1-r6) for continuity and better learning
     - Action: TCP pose (9D: x,y,z + r1-r6) + gripper (1D) = 10D
     - Observation: TCP pose (9D) + gripper (1D) = 10D

   6D Rotation Representation:
   - r1, r2, r3: First column of rotation matrix
   - r4, r5, r6: Second column of rotation matrix
   - Reference: "On the Continuity of Rotation Representations in Neural Networks"

Note: Python API can only use NRT modes due to language timing limitations.

Reference: https://github.com/Dobot-Arm/TCP-IP-Python-V4
"""

import time
from functools import cached_property
from typing import Any, Optional
import re
import threading

from lerobot.robots.bi_dobot_nova5.TCP_IP_Python_V4.dobot_api import DobotApiFeedBack,DobotApiDashboard
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.bi_dobot_nova5.config_bi_dobot_nova5 import ControlMode, BiDobotNova5Config
from lerobot.robots.bi_dobot_nova5.xense_gripper import Gripper
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import (
    get_logger,
    quaternion_to_euler,
    quaternion_to_rotation_6d,
    euler_to_quaternion,
    rotation_6d_to_quaternion,
)

# Alias for dobot_api.Mode for convenience
# Mode = DobotApiDashboard.RequestControl()

# Constants from dobot_api
JOINT_DOF = 6  # Dobot Nova5 robot joint DOF
MM_PER_METER = 1000.0

class DobotNova5(Robot):
    """Dobot Nova5 6-DOF collaborative robot.

    This class implements the LeRobot Robot interface for the Dobot Nova5 robot,
    supporting two control modes for joint space and Cartesian space control.

    Control Modes (NRT only for Python API):
        - JOINT_MOTION: Joint motion control (maps to NRT_JOINT_MOTION)
        - CARTESIAN_MOTION: Cartesian motion control (maps to NRT_CARTESIAN_MOTION)

    Example:
        >>> from lerobot.robots.bi_dobot_nova5.TCP_IP_Python_V4.dobot_api import DobotApiFeedBack,DobotApiDashboard
        >>> from lerobot.robots.bi_dobot_nova5.config_bi_dobot_nova5 import BiDobotNova5Config, ControlMode
        >>> # Joint motion control
        >>> config = BiDobotNova5Config(
        ...     left_robot_ip="192.168.5.101",
        ...     right_robot_ip="192.168.5.102",
        ...     control_mode=ControlMode.JOINT_MOTION,
        ... )
        >>> # Cartesian motion control
        >>> config = BiDobotNova5Config(
        ...     left_robot_ip="192.168.5.101",
        ...     right_robot_ip="192.168.5.102",
        ...     control_mode=ControlMode.CARTESIAN_MOTION,
        ... )
        >>> robot = BiDobotNova5(config)
        >>> robot.connect()
        >>> obs = robot.get_observation()
        >>> robot.send_action({"joint_1.pos": 0.0, ...})
        >>> robot.disconnect()
    """

    config_class = BiDobotNova5Config
    name = "bi_dobot_nova5"

    def __init__(self, config: BiDobotNova5Config):
        super().__init__(config)
        self.config = config

        # Logger
        self.logger = get_logger("BiDobotNova5")

        # Robot interface (initialized on connect)
        # Note: Python API only supports NRT (Non-Real-Time) modes, no Scheduler needed
        self._left_robot: DobotApiDashboard | None = None
        self._right_robot: DobotApiDashboard | None = None
        self._left_feedFour: DobotApiFeedBack | None = None
        self._right_feedFour: DobotApiFeedBack | None = None
        self._left_feedInfo = []
        self._right_feedInfo = []
        self.__left_globalLockValue = threading.Lock()
        self.__right_globalLockValue = threading.Lock()

        class item:
            def __init__(self):
                self.RobotMode = -1     #
                self.robotCurrentCommandID = 0
                self.MessageSize = -1
                self.DigitalInputs =-1
                self.DigitalOutputs = -1
                self.robotCurrentCommandID = -1
                self.tcpPose = [ 258.1406287 ,  -78.59815584, 734.56544667, -171.70077286,    4.8702051,  -84.81002799]
                self.qActual = [ 11.71805878,   23.65214996,  -77.142659,    -27.70232162, -266.17571411,    7.17711411]
                # 自定义添加所需反馈数据

        self.left_feedData = item()  # 定义结构对象
        self.right_feedData = item() 

        self._left_is_connected = False
        self._right_is_connected = False

        self._xense_left_gripper: Gripper | None = None
        self._xense_right_gripper: Gripper | None = None
        if config.use_left_gripper:
            self._xense_left_gripper = Gripper(config.xense_left_gripper)
        if config.use_right_gripper:
            self._xense_right_gripper = Gripper(config.xense_right_gripper)
        if not config.use_left_gripper:
            self._xense_left_gripper = None
        if not config.use_right_gripper:
            self._xense_right_gripper = None

        # Control state - stores the current dobot_api.Mode
        self._left_current_mode = None
        self._right_current_mode = None

        # Home TCP pose - stored after moving to home position
        # Format: [x, y, z, qw, qx, qy, qz] (7D) - SDK format with quaternion
        self._left_home_tcp_pose: np.ndarray | None = None
        self._right_home_tcp_pose: np.ndarray | None = None

        # Gripper key (1D) - always used
        self._left_gripper_key = "left_gripper.pos"
        self._right_gripper_key = "right_gripper.pos"

        # Initialize keys and buffers based on control mode
        if config.control_mode == ControlMode.JOINT_MOTION:
            self._init_joint_mode()
        elif config.control_mode == ControlMode.CARTESIAN_MOTION:
            self._init_cartesian_mode()
        else:
            raise ValueError(f"Unsupported control_mode: {config.control_mode}")

        self.cameras = make_cameras_from_configs(config.cameras)
        np.set_printoptions(precision=6, suppress=True)

    def _init_joint_mode(self) -> None:
        """Initialize keys and buffers for joint motion control mode."""
        # Joint state observation keys: joint_{1-6}.{pos, vel, effort}
        self._left_joint_pos_keys = tuple(f"left_joint_{i}.pos" for i in range(1, JOINT_DOF + 1))
        self._right_joint_pos_keys = tuple(f"right_joint_{i}.pos" for i in range(1, JOINT_DOF + 1))

        # Joint action keys: joint_{1-6}.pos
        self._left_action_joint_keys = tuple(f"left_joint_{i}.pos" for i in range(1, JOINT_DOF + 1))
        self._right_action_joint_keys = tuple(f"right_joint_{i}.pos" for i in range(1, JOINT_DOF + 1))

        # Pre-cache config values as lists (for API calls)
        self._control_frequency = self.config.control_frequency
        self._aheadtime = self.config.aheadtime
        self._gain = self.config.gain

    def _init_cartesian_mode(self) -> None:
        """Initialize keys and buffers for CARTESIAN_MOTION control mode.

        Uses 6D rotation representation (r1-r6) instead of quaternion for:
        - Continuity: No discontinuities like Euler angles (gimbal lock)
        - No double-cover: Unlike quaternions where q and -q represent same rotation
        - Better for neural networks: Continuous representation is easier to learn

        Reference: "On the Continuity of Rotation Representations in Neural Networks"
        """
        # TCP pose observation/action keys: tcp.{x, y, z, r1, r2, r3, r4, r5, r6}
        # 6D rotation: r1-r3 = first column, r4-r6 = second column of rotation matrix
        self._left_tcp_pose_keys = (
            "left_tcp.x",
            "left_tcp.y",
            "left_tcp.z",
            "left_tcp.r1",
            "left_tcp.r2",
            "left_tcp.r3",
            "left_tcp.r4",
            "left_tcp.r5",
            "left_tcp.r6",
        )
        self._right_tcp_pose_keys = (
            "right_tcp.x",
            "right_tcp.y",
            "right_tcp.z",
            "right_tcp.r1",
            "right_tcp.r2",
            "right_tcp.r3",
            "right_tcp.r4",
            "right_tcp.r5",
            "right_tcp.r6",
        )

        # TCP pose action keys (same as observation keys for 6D rotation)
        self._left_action_tcp_pose_keys = self._left_tcp_pose_keys
        self._right_action_tcp_pose_keys = self._right_tcp_pose_keys

        # Pre-cache config values as lists (for API calls)
        self._control_frequency = self.config.control_frequency
        self._aheadtime = self.config.aheadtime
        self._gain = self.config.gain

    @property
    def _action_ft(self) -> dict[str, type]:
        """Return action features based on control_mode.

        Action space (all include gripper):
        - JOINT_MOTION: joint positions (7D) + gripper (1D) = 8D
        - CARTESIAN_MOTION: TCP pose (9D: xyz + 6D rotation) + gripper (1D) = 10D
        """
        features = {}

        if self.config.control_mode == ControlMode.JOINT_MOTION:
            # Joint positions (7D)
            features.update(dict.fromkeys(self._left_action_joint_keys, float))
            features.update(dict.fromkeys(self._right_action_joint_keys, float))

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            # TCP pose (9D: xyz + 6D rotation)
            features.update(dict.fromkeys(self._left_action_tcp_pose_keys, float))
            features.update(dict.fromkeys(self._right_action_tcp_pose_keys, float))

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Always include gripper (1D)
        features[self._left_gripper_key] = float
        features[self._right_gripper_key] = float
        return features

    @property
    def _proprioception_ft(self) -> dict[str, type]:
        """Return observation features based on control_mode.

        Observation space (all include gripper):
        - JOINT_MOTION: joint pos (7D) + gripper pos (1D) = 8D
        - CARTESIAN_MOTION: TCP pose (9D: xyz + 6D rotation) + gripper (1D) = 10D
        """
        features = {}

        if self.config.control_mode == ControlMode.JOINT_MOTION:
            # Joint positions (7D)
            features.update(dict.fromkeys(self._left_joint_pos_keys, float))
            features.update(dict.fromkeys(self._right_joint_pos_keys, float))

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            # TCP pose (9D: xyz + 6D rotation)
            features.update(dict.fromkeys(self._left_tcp_pose_keys, float))
            features.update(dict.fromkeys(self._right_tcp_pose_keys, float))

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Always include gripper position (1D)
        features[self._left_gripper_key] = float
        features[self._right_gripper_key] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Return camera/image features from Xense Gripper and external cameras."""
        features = {}

        for cam_name in self.cameras:
            features[cam_name] = (
                self.config.cameras[cam_name].height,
                self.config.cameras[cam_name].width,
                3,
            )

        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Return observation features (robot states + cameras)."""
        return {**self._proprioception_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Return action features based on control mode."""
        return self._action_ft

    @property
    def is_connected(self) -> bool:
        return (
            self._left_is_connected
            and self._right_is_connected
            and self._left_robot is not None
            and self._right_robot is not None
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def is_calibrated(self) -> bool:
        """Dobot Nova5 robots are factory calibrated."""
        return self.is_connected

    def calibrate(self) -> None:
        """Dobot Nova5 robots are factory calibrated, no runtime calibration needed."""
        self.logger.info("Dobot Nova5 is factory calibrated, no runtime calibration needed.")


    def configure(self) -> None:
        """Configure the robot based on control mode."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info(f"Configuring robot for {self.config.control_mode.value} mode...")

    def _parse_dobot_response(self, value_recv: str) -> tuple[int, list[str]]:
        """Parse Dobot TCP response: ErrorID,{ResultID or values},Command(...);"""
        if not isinstance(value_recv, str):
            raise RuntimeError(f"Invalid Dobot response type: {type(value_recv).__name__}")

        if "Not Tcp" in value_recv:
            raise RuntimeError("Robot is not in TCP control mode")

        match = re.match(r"\s*(-?\d+)\s*,\s*\{([^}]*)\}", value_recv)
        if match is None:
            raise RuntimeError(f"Could not parse Dobot response: {value_recv!r}")

        error_id = int(match.group(1))
        values = [value.strip() for value in match.group(2).split(",") if value.strip()]
        return error_id, values

    def _dobot_error_detail(self) -> str:
        if self._left_robot is None or self._right_robot is None:
            return ""
        try:
            return self._left_robot.GetErrorID().strip() + " " + self._right_robot.GetErrorID().strip()
        except Exception as e:
            return f"failed to read GetErrorID: {e}"

    def _raise_if_dobot_error(self, robot: DobotApiDashboard, response: str, command_name: str) -> list[str]:
        error_id, values = self._parse_dobot_response(response)
        if error_id == 0:
            return values

        message = f"{command_name} failed with ErrorID {error_id}: {response.strip()}"
        error_detail = self._dobot_error_detail()
        if error_detail:
            message = f"{message}; GetErrorID: {error_detail}"
        raise RuntimeError(message)

    def _wait_for_command(self, command_id: int, description: str, timeout_s: float = 60.0) -> None:
        start_time = time.time()
        last_log_time = 0.0
        command_id = int(command_id)

        while True:
            left_robot_mode = int(self.left_feedData.RobotMode)
            left_current_command_id = int(self.left_feedData.robotCurrentCommandID)
            right_robot_mode = int(self.right_feedData.RobotMode)
            right_current_command_id = int(self.right_feedData.robotCurrentCommandID)
            if left_robot_mode == 9 or right_robot_mode == 9:
                raise RuntimeError(
                    f"{description} failed: robot entered error mode while waiting for command "
                    f"{command_id}. GetErrorID: {self._dobot_error_detail()}"
                )

            if left_robot_mode == 5 and left_current_command_id == command_id and right_robot_mode == 5 and right_current_command_id == command_id:
                return

            now = time.time()
            if now - start_time > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for {description}: target command ID={command_id}, "
                    f"current command ID={left_current_command_id}, RobotMode={left_robot_mode}"
                    f"current command ID={right_current_command_id}, RobotMode={right_robot_mode}"
                )

            if now - last_log_time >= 2.0:
                self.logger.info(
                    f"Waiting for {description}: RobotMode={left_robot_mode}, "
                    f"CurrentCommandId={left_current_command_id}, target={command_id}",
                    f"RobotMode={right_robot_mode}, "
                    f"CurrentCommandId={right_current_command_id}, target={command_id}"
                )
                last_log_time = now

            time.sleep(0.1)

    def _wait_for_joint_target(
        self,
        target_joint_deg: list[float],
        description: str,
        tolerance_deg: float = 0.5,
        timeout_s: float = 60.0,
    ) -> None:
        target = np.asarray(target_joint_deg, dtype=np.float64)
        start_time = time.time()
        last_log_time = 0.0

        while True:
            left_robot_mode = int(self.left_feedData.RobotMode)
            right_robot_mode = int(self.right_feedData.RobotMode)
            left_current_joint = np.asarray(self.left_feedData.qActual, dtype=np.float64)
            right_current_joint = np.asarray(self.right_feedData.qActual, dtype=np.float64)
            left_max_abs_error = float(np.max(np.abs(left_current_joint - target)))
            right_max_abs_error = float(np.max(np.abs(right_current_joint - target)))

            if left_robot_mode == 9 or right_robot_mode == 9:
                raise RuntimeError(
                    f"{description} failed: left robot entered error mode. "
                    f"GetErrorID: {self._dobot_error_detail()}"
                )

            if left_robot_mode == 5 and left_max_abs_error <= tolerance_deg and right_robot_mode == 5 and right_max_abs_error <= tolerance_deg:
                return

            now = time.time()
            if now - start_time > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for {description}: left max_joint_error={left_max_abs_error:.3f} deg, right max_joint_error={right_max_abs_error:.3f} deg, "
                    f"RobotMode={left_robot_mode}, target={target.tolist()}, current={left_current_joint.tolist()}",
                    f"RobotMode={right_robot_mode}, target={target.tolist()}, current={right_current_joint.tolist()}"
                )

            if now - last_log_time >= 2.0:
                self.logger.info(
                    f"Waiting for {description}: RobotMode={left_robot_mode}, "
                    f"max_joint_error={left_max_abs_error:.3f} deg",
                    f"RobotMode={right_robot_mode}, target={target.tolist()}, current={right_current_joint.tolist()}"
                )
                last_log_time = now

            time.sleep(0.1)

    def _wait_for_first_feedback(self, timeout_s: float = 3.0) -> None:
        start_time = time.time()
        while self.left_feedData.MessageSize == -1 and self.left_feedData.RobotMode == -1 and self.right_feedData.MessageSize == -1 and self.right_feedData.RobotMode == -1:
            if time.time() - start_time > timeout_s:
                return
            time.sleep(0.02)

    def _wait_until_not_error_mode(self, timeout_s: float = 10.0) -> None:
        start_time = time.time()
        while int(self.left_feedData.RobotMode) == 9 and int(self.right_feedData.RobotMode) == 9:
            if time.time() - start_time > timeout_s:
                raise TimeoutError(
                    f"Robot stays in error mode (RobotMode=9) after ClearError. "
                    f"GetErrorID: {self._dobot_error_detail()}"
                )
            time.sleep(0.1)

    def _send_movj_joint_v4(self, robot: DobotApiDashboard, joint_degrees: list[float]) -> str:
        if robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        # V4 standard API: MovJ with coordinateMode=1 (joint target).
        # Use per-command velocity scale from config.start_vel_scale.
        target = [float(value) for value in joint_degrees]
        return robot.MovJ(
            target[0],
            target[1],
            target[2],
            target[3],
            target[4],
            target[5],
            1,  # coordinateMode=1 -> joint
            v=int(self.config.start_vel_scale),
        )

    def _move_joint_movj(self, robot: DobotApiDashboard, joint_degrees: list[float], description: str) -> None:
        if robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        target = [float(value) for value in joint_degrees]
        self._raise_if_dobot_error(
            robot.SpeedFactor(int(self.config.start_vel_scale)),
            "SpeedFactor",
        )
        response = self._send_movj_joint_v4(robot, target)
        self.logger.info(f"{description} MovJ(joint) response: {response.strip()}")
        values = self._raise_if_dobot_error(robot, response, "MovJ(joint)")
        command_name = "MovJ(joint)"

        if values:
            try:
                command_id = int(float(values[0]))
                self._wait_for_command(command_id, description)
                return
            except (ValueError, TypeError):
                self.logger.warn(
                    f"{command_name} response did not provide a valid command ID ({values}), "
                    "falling back to joint-error completion check."
                )

        self._wait_for_joint_target(target, description)

    def _initialize_gripper_position(self) -> None:
        if self._xense_left_gripper is None or self._xense_right_gripper is None:
            return

        if self.config.left_gripper_init_open:
            self._xense_left_gripper._gripper.set_position_sync(
                self.config.left_gripper_max_pos,
                vmax=self.config.left_gripper_velocity,
                fmax=self.config.left_gripper_force,
            )
        else:
            self._xense_left_gripper._gripper.set_position_sync(
                self.config.left_gripper_min_pos,
                vmax=self.config.left_gripper_velocity,
                fmax=self.config.left_gripper_force,
            )
        if self.config.right_gripper_init_open:
            self._xense_right_gripper._gripper.set_position_sync(
                self.config.right_gripper_max_pos,
                vmax=self.config.right_gripper_velocity,
                fmax=self.config.right_gripper_force,
            )
        else:
            self._xense_right_gripper._gripper.set_position_sync(
                self.config.right_gripper_min_pos,
                vmax=self.config.right_gripper_velocity,
                fmax=self.config.right_gripper_force,
            )

    def _current_gripper_position(self) -> float:
        if self._xense_left_gripper and self.config.use_left_gripper:
            return float(self._xense_left_gripper.get_gripper_position())
        if self._xense_right_gripper and self.config.use_right_gripper:
            return float(self._xense_right_gripper.get_gripper_position())
        return 0.0

    def _current_tcp_pose_quat_from_feedback(self) -> np.ndarray:
        left_tcp_pose = np.asarray(self.left_feedData.tcpPose, dtype=np.float64)
        right_tcp_pose = np.asarray(self.right_feedData.tcpPose, dtype=np.float64)
        # Dobot feedback uses mm for xyz. Convert to meters for teleop stack.
        left_pos_m = left_tcp_pose[:3] / MM_PER_METER
        left_quat = euler_to_quaternion(
            np.deg2rad(left_tcp_pose[3]),
            np.deg2rad(left_tcp_pose[4]),
            np.deg2rad(left_tcp_pose[5]),
        )
        left_pose = np.array(
            [left_pos_m[0], left_pos_m[1], left_pos_m[2], left_quat[0], left_quat[1], left_quat[2], left_quat[3]],
            dtype=np.float32,
        )
        right_pos_m = right_tcp_pose[:3] / MM_PER_METER
        right_quat = euler_to_quaternion(
            np.deg2rad(right_tcp_pose[3]),
            np.deg2rad(right_tcp_pose[4]),
            np.deg2rad(right_tcp_pose[5]),
        )
        right_pose = np.array(
            [right_pos_m[0], right_pos_m[1], right_pos_m[2], right_quat[0], right_quat[1], right_quat[2], right_quat[3]],
            dtype=np.float32,
        )
        return left_pose, right_pose

    def connect(self, calibrate: bool = False, go_to_start: bool = True) -> None:
        """Connect to the Dobot Nova5 robot.

        Args:
            calibrate: Ignored (Dobot Nova5 robots are factory calibrated)
            go_to_start: If provided, overrides config.go_to_start. If None, uses config.go_to_start.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(
                f"{self} already connected, do not run `robot.connect()` twice."
            )

        try:
            self.logger.info(f"Connecting to Dobot Nova5 robot: {self.config.left_robot_ip}")
            self.logger.info(f"Connecting to Dobot Nova5 robot: {self.config.right_robot_ip}")

            # Create robot interface
            self._left_robot = DobotApiDashboard(self.config.left_robot_ip, self.config.left_dashboardPort)
            self._left_feedFour = DobotApiFeedBack(self.config.left_robot_ip, self.config.left_feedPortFour)
            self._right_robot = DobotApiDashboard(self.config.right_robot_ip, self.config.right_dashboardPort)
            self._right_feedFour = DobotApiFeedBack(self.config.right_robot_ip, self.config.right_feedPortFour)

            # Start feedback thread to continuously read robot state (for get_observation)
            # TODO: Needs testing
            feed_thread = threading.Thread(
                target=self.get_robot_state)  # robot state feedback thread
            feed_thread.daemon = True
            feed_thread.start()
            self._wait_for_first_feedback()

            # Clear fault before enabling if needed.
            if int(self.left_feedData.RobotMode) == 9 and int(self.right_feedData.RobotMode) == 9:
                self.logger.warn("Robot is in error mode before enabling, trying ClearError ...")
                self._raise_if_dobot_error(self._left_robot.ClearError(), "ClearError")
                self._raise_if_dobot_error(self._right_robot.ClearError(), "ClearError")
                self._wait_until_not_error_mode()
                self.logger.info("Fault on the connected robot is cleared")

            # Enable the robot
            self.logger.info("Enabling robot...")
            if self.config.control_mode in (ControlMode.CARTESIAN_MOTION, ControlMode.JOINT_MOTION):
                left_enable_response = self._left_robot.EnableRobot()
                right_enable_response = self._right_robot.EnableRobot()
                left_enable_error, _ = self._parse_dobot_response(left_enable_response)
                right_enable_error, _ = self._parse_dobot_response(right_enable_response)
                if left_enable_error != 0 or right_enable_error != 0:
                    left_mode_response = self._left_robot.RobotMode()
                    right_mode_response = self._right_robot.RobotMode()
                    left_mode_error, left_mode_values = self._parse_dobot_response(left_mode_response)
                    right_mode_error, right_mode_values = self._parse_dobot_response(right_mode_response)
                    left_current_mode = int(float(left_mode_values[0])) if left_mode_error == 0 and left_mode_values else -1
                    right_current_mode = int(float(right_mode_values[0])) if right_mode_error == 0 and right_mode_values else -1
                    if left_current_mode in (5, 6, 7, 8) and right_current_mode in (5, 6, 7, 8):
                        self.logger.warn(
                            f"EnableRobot returned {left_enable_error}, but RobotMode={left_current_mode}. "
                            f"EnableRobot returned {right_enable_error}, but RobotMode={right_current_mode}. "
                            "Proceeding with existing enabled/control state."
                        )
                    else:
                        raise RuntimeError(
                            f"EnableRobot failed with ErrorID {left_enable_error}: {left_enable_response.strip()} "
                            f"EnableRobot failed with ErrorID {right_enable_error}: {right_enable_response.strip()} "
                            f"(RobotMode={left_current_mode}); GetErrorID: {self._dobot_error_detail()}"
                            f"(RobotMode={right_current_mode}); GetErrorID: {self._dobot_error_detail()}"
                        )
                else:
                    if self.config.control_mode == ControlMode.CARTESIAN_MOTION:
                        self.logger.info("Robot TCP enabled successfully.")
                    else:
                        self.logger.info("Robot Joint enabled successfully.")
            else:
                raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

            if self.left_feedData.RobotMode == 5 and self.right_feedData.RobotMode == 5:
                self.logger.info("Robot is enabled and ready.")
            else:
                self.logger.warn(
                    f"Robot is in unexpected mode {self.left_feedData.RobotMode} after enabling. Check robot status."
                    f"Robot is in unexpected mode {self.right_feedData.RobotMode} after enabling. Check robot status."
                )

            # Wait for robot to become operational
            timeout = 30  # seconds
            start_time = time.time()
            while not self.left_feedData.RobotMode == 5 and self.right_feedData.RobotMode == 5:  # ROBOT_MODE_ENABLE  Enabled and idle
                if time.time() - start_time > timeout:
                    raise RuntimeError(f"Robot did not become operational within {timeout} seconds")
                time.sleep(0.1)

            self.logger.info("Robot is now operational.")

            # Connect Xense Gripper end-effector (provides gripper + wrist_cam + tactile)
            if self._xense_left_gripper and self.config.use_left_gripper:
                self.logger.info("Connecting Xense Gripper...")
                self._xense_left_gripper.connect()
            if self._xense_right_gripper and self.config.use_right_gripper:
                self.logger.info("Connecting Xense Gripper...")
                self._xense_right_gripper.connect()

            # Connect external cameras (e.g., scene cameras)
            for cam in self.cameras.values():
                cam.connect()

            # Set _is_connected to True before calling methods that check is_connected
            self._is_connected = True

            # Move to start position if requested (use parameter if provided, otherwise use config)
            self.config.go_to_start = go_to_start if go_to_start is not None else self.config.go_to_start
            if self.config.go_to_start:
                self._go_to_start()

            # Configure control parameters
            self.configure()

            mode_desc = self.config.control_mode.value
            if self._xense_left_gripper and self.config.use_left_gripper:
                gripper_status = "with Xense Left Gripper (gripper + wrist_cam + tactile)"
            if self._xense_right_gripper and self.config.use_right_gripper:
                gripper_status = "with Xense Right Gripper (gripper + wrist_cam + tactile)"
            else:
                gripper_status = "no gripper"
            self.logger.info(f"✅ Dobot Nova5 connected and ready in {mode_desc} mode ({gripper_status}).")

        except Exception as e:
            self.logger.error(f"Failed to connect to Dobot Nova5 robot: {e}")
            self._robot = None
            self._is_connected = False
            raise

    def _go_to_home(self) -> None:
        """Move robot to home position using MoveJ primitive.

        Uses ExecutePrimitive("MoveJ") to move to factory-defined home pose:
        - target: [0.0, -40.0, 0.0, 90.0, 0.0, 40.0, 0.0] degrees
        - jntVelScale: 20
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Moving to home position...")

        self._move_joint_movj(self._left_robot, self.config.left_home_point_list, "left home position")
        self._move_joint_movj(self._right_robot, self.config.right_home_point_list, "right home position")
        self._initialize_gripper_position()
        self._left_home_tcp_pose, self._right_home_tcp_pose = self._current_tcp_pose_quat_from_feedback()
        self.logger.info(f"Left home TCP pose: {self._left_home_tcp_pose}")
        self.logger.info(f"Right home TCP pose: {self._right_home_tcp_pose}")
        self.logger.info("✅ Robot at home position.")
        if self._xense_left_gripper is not None and self._xense_right_gripper is not None:
            self.logger.info(f"Left gripper position: {self._xense_left_gripper.get_gripper_position()}")
            self.logger.info(f"Right gripper position: {self._xense_right_gripper.get_gripper_position()}")

    def _go_to_start(self) -> None:
        """Move robot to start position using MoveJ primitive.

        Uses ExecutePrimitive("MoveJ") with configurable parameters:
        - jntVelScale: Joint velocity scale 1-100 (from config.start_vel_scale)
        - target: Start joint position in degrees (from config.start_position_degree)
        """
        if not self._is_connected or self._left_robot is None or self._right_robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Moving to start position...")

        self._move_joint_movj(self._left_robot, self.config.left_start_position_degree, "left start position")
        self._move_joint_movj(self._right_robot, self.config.right_start_position_degree, "right start position")
        self._initialize_gripper_position()
        self.logger.info("✅ Robot at start position.")
        if self._xense_left_gripper is not None and self._xense_right_gripper is not None:
            self.logger.info(f"Left gripper position: {self._xense_left_gripper.get_gripper_position()}")
            self.logger.info(f"Right gripper position: {self._xense_right_gripper.get_gripper_position()}")

    def reset_to_initial_position(self) -> None:
        """Reset robot to initial position based on config.go_to_start.

        If config.go_to_start=True, calls _go_to_start().
        Otherwise, calls _go_to_home().
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.go_to_start:
            self.logger.info("Resetting to start position (config.go_to_start=True)")
            self._go_to_start()
        else:
            self.logger.info("Resetting to home position (config.go_to_start=False)")
            self._go_to_home()

    def get_robot_state(self) -> Optional[dict]:
        # 获取机器人状态
        while True:
            if self._left_feedFour is None or self._right_feedFour is None:
                return None
            try:
                left_feedInfo = self._left_feedFour.feedBackData()
                right_feedInfo = self._right_feedFour.feedBackData()
            except Exception:
                # Socket may be closing during disconnect; exit feedback thread quietly.
                return None
            with self.__globalLockValue:
                if left_feedInfo is not None and right_feedInfo is not None:   
                    if hex((left_feedInfo['TestValue'][0])) == '0x123456789abcdef' and hex((right_feedInfo['TestValue'][0])) == '0x123456789abcdef':
                        # 基础字段
                        self.left_feedData.MessageSize = left_feedInfo['len'][0]
                        self.left_feedData.RobotMode = left_feedInfo['RobotMode'][0]
                        self.left_feedData.DigitalInputs = left_feedInfo['DigitalInputs'][0]
                        self.left_feedData.DigitalOutputs = left_feedInfo['DigitalOutputs'][0]
                        self.left_feedData.robotCurrentCommandID = left_feedInfo['CurrentCommandId'][0]
                        self.left_feedData.tcpPose = left_feedInfo['ToolVectorActual'][0]
                        self.left_feedData.qActual = left_feedInfo['QActual'][0]
                        self.right_feedData.MessageSize = right_feedInfo['len'][0]
                        self.right_feedData.RobotMode = right_feedInfo['RobotMode'][0]
                        self.right_feedData.DigitalInputs = right_feedInfo['DigitalInputs'][0]
                        self.right_feedData.DigitalOutputs = right_feedInfo['DigitalOutputs'][0]
                        self.right_feedData.robotCurrentCommandID = right_feedInfo['CurrentCommandId'][0]
                        self.right_feedData.tcpPose = right_feedInfo['ToolVectorActual'][0]
                        self.right_feedData.qActual = right_feedInfo['QActual'][0]

    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation based on control_mode.

        Returns a dictionary with observation data. The content depends on control_mode:
        - JOINT_MOTION: joint_1-7.{pos,vel,effort} (21D) + gripper.pos (1D) = 22D
        - CARTESIAN_MOTION: tcp.{x,y,z,r1-r6} (9D) + gripper (1D) = 10D

        Also includes camera images if configured.
        """
        if not self.is_connected or self._left_robot is None or self._right_robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        if self.config.control_mode == ControlMode.JOINT_MOTION:
            # Joint positions (7D)
            for i, key in enumerate(self._left_joint_pos_keys):
                obs_dict[key] = self.left_feedData.qActual[i]
            for i, key in enumerate(self._right_joint_pos_keys):
                obs_dict[key] = self.right_feedData.qActual[i]

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            # TCP pose from SDK: [x, y, z, qw, qx, qy, qz]
            # TODO
            left_tcp_pose = self.left_feedData.tcpPose
            right_tcp_pose = self.right_feedData.tcpPose

            # Position (3D)
            obs_dict["left_tcp.x"] = left_tcp_pose[0] / MM_PER_METER
            obs_dict["left_tcp.y"] = left_tcp_pose[1] / MM_PER_METER
            obs_dict["left_tcp.z"] = left_tcp_pose[2] / MM_PER_METER
            obs_dict["right_tcp.x"] = right_tcp_pose[0] / MM_PER_METER
            obs_dict["right_tcp.y"] = right_tcp_pose[1] / MM_PER_METER
            obs_dict["right_tcp.z"] = right_tcp_pose[2] / MM_PER_METER

            # Convert quaternion to 6D rotation representation
            left_quat = euler_to_quaternion(np.deg2rad(left_tcp_pose[3]), np.deg2rad(left_tcp_pose[4]), np.deg2rad(left_tcp_pose[5]))
            right_quat = euler_to_quaternion(np.deg2rad(right_tcp_pose[3]), np.deg2rad(right_tcp_pose[4]), np.deg2rad(right_tcp_pose[5]))
            left_r6d = quaternion_to_rotation_6d(left_quat[0], left_quat[1], left_quat[2], left_quat[3])
            right_r6d = quaternion_to_rotation_6d(right_quat[0], right_quat[1], right_quat[2], right_quat[3])

            obs_dict["left_tcp.r1"] = left_r6d[0]
            obs_dict["left_tcp.r2"] = left_r6d[1]
            obs_dict["left_tcp.r3"] = left_r6d[2]
            obs_dict["left_tcp.r4"] = left_r6d[3]
            obs_dict["left_tcp.r5"] = left_r6d[4]
            obs_dict["left_tcp.r6"] = left_r6d[5]
            obs_dict["right_tcp.r1"] = right_r6d[0]
            obs_dict["right_tcp.r2"] = right_r6d[1]
            obs_dict["right_tcp.r3"] = right_r6d[2]
            obs_dict["right_tcp.r4"] = right_r6d[3]
            obs_dict["right_tcp.r5"] = right_r6d[4]
            obs_dict["right_tcp.r6"] = right_r6d[5]


        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Get data from Xense Gripper (gripper + tactile sensors)
        if self._xense_left_gripper is not None and self.config.use_left_gripper:
            # Read sensors (keys are mapped from SN to sensor_keys names)
            sensor_data = self._xense_left_gripper.get_sensor_data()
            for key, data in sensor_data.items():
                obs_dict[key] = data

            # Read gripper position
            obs_dict[self._left_gripper_key] = self._xense_left_gripper.get_gripper_position()
            obs_dict[self._right_gripper_key] = self._xense_right_gripper.get_gripper_position()

        # External camera observations (scene cameras, etc.)
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def forward_kinematics(self, joint_positions_deg: list | np.ndarray) -> np.ndarray:
        """Compute forward kinematics: joint positions -> TCP pose.
        
        This is a pure mathematical computation that only requires the Robot object
        to be created (for accessing kinematic model), not necessarily connected/enabled.
        
        Args:
            joint_positions_deg: Joint positions in degrees (7D)
            
        Returns:
            TCP pose [x, y, z, qw, qx, qy, qz] (7D)
        """
        if self._robot is None:
            raise RuntimeError("Robot object not created. Call connect() first or create Robot manually.")
        
        # V4 API PositiveKin returns TCP pose in controller format:
        # [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg].
        response = self._robot.PositiveKin(
            float(joint_positions_deg[0]),
            float(joint_positions_deg[1]),
            float(joint_positions_deg[2]),
            float(joint_positions_deg[3]),
            float(joint_positions_deg[4]),
            float(joint_positions_deg[5]),
        )
        values = self._raise_if_dobot_error(response, "PositiveKin")
        if len(values) < 6:
            raise RuntimeError(
                f"PositiveKin response missing pose values: {response.strip()}"
            )

        x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = [float(v) for v in values[:6]]
        quat = euler_to_quaternion(
            np.deg2rad(rx_deg),
            np.deg2rad(ry_deg),
            np.deg2rad(rz_deg),
        )
        return np.array(
            [
                x_mm / MM_PER_METER,
                y_mm / MM_PER_METER,
                z_mm / MM_PER_METER,
                quat[0],
                quat[1],
                quat[2],
                quat[3],
            ],
            dtype=np.float32,
        )

    def get_start_tcp_pose_euler(self) -> np.ndarray:
        """Get TCP pose at start position in Euler format [x, y, z, roll, pitch, yaw, gripper_pos].
        
        This can be called after Robot object is created, even before robot is enabled.
        
        Returns:
            numpy array of shape (7,) with [x, y, z, roll, pitch, yaw, gripper_pos]
        """
        left_tcp_pose_quat = self.forward_kinematics(self.config.left_start_position_degree)
        right_tcp_pose_quat = self.forward_kinematics(self.config.right_start_position_degree)
        
        # Convert quaternion to Euler angles
        left_euler = quaternion_to_euler(
            left_tcp_pose_quat[3], left_tcp_pose_quat[4], left_tcp_pose_quat[5], left_tcp_pose_quat[6]
        )
        right_euler = quaternion_to_euler(
            right_tcp_pose_quat[3], right_tcp_pose_quat[4], right_tcp_pose_quat[5], right_tcp_pose_quat[6]
        )
        
        # Get initial gripper position based on config
        left_gripper_pos = self.config.left_gripper_max_pos if self.config.left_gripper_init_open else 0.0
        right_gripper_pos = self.config.right_gripper_max_pos if self.config.right_gripper_init_open else 0.0
        
        left_tcp_pose = np.array(
            [left_tcp_pose_quat[0], left_tcp_pose_quat[1], left_tcp_pose_quat[2], left_euler[0], left_euler[1], left_euler[2], left_gripper_pos],
            dtype=np.float32,
        )
        right_tcp_pose = np.array(
            [right_tcp_pose_quat[0], right_tcp_pose_quat[1], right_tcp_pose_quat[2], right_euler[0], right_euler[1], right_euler[2], right_gripper_pos],
            dtype=np.float32,
        )
        return left_tcp_pose, right_tcp_pose

    def get_current_tcp_pose_euler(self) -> np.ndarray:
        """Get current TCP pose in Euler angles format [x, y, z, roll, pitch, yaw, gripper_pos].

        This method can be used for getting the current TCP pose in Euler angles format for initializing teleoperators (e.g., spacemouse) with the robot's
        current TCP pose. Only available in CARTESIAN_MOTION mode.

        Returns:
            numpy array of shape (7,) with [x, y, z, roll, pitch, yaw, gripper_pos]
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode != ControlMode.CARTESIAN_MOTION:
            raise ValueError("get_current_tcp_pose_euler requires CARTESIAN_MOTION mode")

        left_tcp_pose = np.asarray(self.left_feedData.tcpPose, dtype=np.float32)
        right_tcp_pose = np.asarray(self.right_feedData.tcpPose, dtype=np.float32)
        left_gripper_pos = self._current_gripper_position()
        right_gripper_pos = self._current_gripper_position()

        return np.array(
            [
                left_tcp_pose[0] / MM_PER_METER,
                left_tcp_pose[1] / MM_PER_METER,
                left_tcp_pose[2] / MM_PER_METER,
                np.deg2rad(left_tcp_pose[3]),
                np.deg2rad(left_tcp_pose[4]),
                np.deg2rad(left_tcp_pose[5]),
                left_gripper_pos,
                right_tcp_pose[0] / MM_PER_METER,
                right_tcp_pose[1] / MM_PER_METER,
                right_tcp_pose[2] / MM_PER_METER,
                np.deg2rad(right_tcp_pose[3]),
                np.deg2rad(right_tcp_pose[4]),
                np.deg2rad(right_tcp_pose[5]),
                right_gripper_pos,
            ],
            dtype=np.float32,
        )

    def get_current_tcp_pose_quat(self) -> np.ndarray:
        """Get current TCP pose in quaternion format [x, y, z, qw, qx, qy, qz, gripper_pos].

        This method can be used for getting the current TCP pose in quaternion format for initializing teleoperators (e.g., pico4) with the robot's
        current TCP pose. Only available in CARTESIAN_MOTION mode.

        Returns:
            numpy array of shape (8,) with [x, y, z, qw, qx, qy, qz, gripper_pos]
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.control_mode != ControlMode.CARTESIAN_MOTION:
            raise ValueError("get_current_tcp_pose_quat requires CARTESIAN_MOTION mode")

        left_tcp_pose, right_tcp_pose = self._current_tcp_pose_quat_from_feedback()
        left_gripper_pos = self._current_gripper_position()
        right_gripper_pos = self._current_gripper_position()
        left_pose = np.array(
            [left_tcp_pose[0], left_tcp_pose[1], left_tcp_pose[2], left_tcp_pose[3], left_tcp_pose[4], left_tcp_pose[5], left_tcp_pose[6], left_gripper_pos],
            dtype=np.float32,
        )
        right_pose = np.array(
            [right_tcp_pose[0], right_tcp_pose[1], right_tcp_pose[2], right_tcp_pose[3], right_tcp_pose[4], right_tcp_pose[5], right_tcp_pose[6], right_gripper_pos],
            dtype=np.float32,
        )
        return left_pose, right_pose

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action command to the robot.

        The action format depends on the control_mode and use_force:
        - JOINT_MOTION: {joint_i.pos: float} for i in 1..7, + gripper.pos
        - CARTESIAN_MOTION + use_force=False: {tcp.x, tcp.y, tcp.z, tcp.r1-r6} + gripper.pos
        - CARTESIAN_MOTION + use_force=True: pose (9D) + wrench (6D) + gripper.pos

        Args:
            action: Dictionary of action values

        Returns:
            The action that was actually sent (may be clipped for safety)
        """
        if not self.is_connected or self._left_robot is None or self._right_robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Check for fault
        if self.left_feedData.RobotMode == 9 or self.right_feedData.RobotMode == 9:
            raise RuntimeError(f"Left robot fault detected. GetErrorID: {self._left_dobot_error_detail()}")

        # Send robot arm action
        if self.config.control_mode == ControlMode.JOINT_MOTION:
            result = self._send_joint_position_action(action)

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            result = self._send_cartesian_pure_motion_action(action)

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Send gripper action
        self._send_gripper_action(action)

        return result

    def _send_joint_position_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send joint command.

        Uses SendJointPosition with motion constraints (max_vel, max_acc) from config.
        API signature: SendJointPosition(target_pos, target_vel, max_vel, max_acc)

        Note: target_vel is always [0.0] * DoF for impedance control.

        Action keys: action.joint_{1-7}.pos
        """
        # Extract target positions directly from action
        target_pos = [action[key] for key in self._action_joint_keys]

        # Send command using API
        response = self._robot.ServoJ(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            target_pos[3],
            target_pos[4],
            target_pos[5],
            1.0 / self._control_frequency,
            self._aheadtime,
            self._gain,
        )
        self._raise_if_dobot_error(response, "ServoJ")

        return action

    def _send_cartesian_pure_motion_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send Cartesian pure motion command (NRT mode, no force control).

        Action keys: action.tcp.{x,y,z,r1,r2,r3,r4,r5,r6}

        The action uses 6D rotation representation, then converts it to Dobot's
        ServoP Cartesian pose format [x, y, z, rx, ry, rz].
        """
        # Extract position
        x_m, y_m, z_m = action["tcp.x"], action["tcp.y"], action["tcp.z"]
        # LeRobot actions use meters; Dobot ServoP expects millimeters.
        x_mm = float(x_m) * MM_PER_METER
        y_mm = float(y_m) * MM_PER_METER
        z_mm = float(z_m) * MM_PER_METER

        # Extract 6D rotation and convert to quaternion
        r6d = np.array(
            [
                action["tcp.r1"],
                action["tcp.r2"],
                action["tcp.r3"],
                action["tcp.r4"],
                action["tcp.r5"],
                action["tcp.r6"],
            ]
        )
        quat = rotation_6d_to_quaternion(r6d)  # Returns [qw, qx, qy, qz]
        # Convert quaternion to Euler angles
        euler = quaternion_to_euler(
            quat[0], quat[1], quat[2], quat[3]
        )
        self.logger.debug(
            f"ServoP xyz: [{x_m:.6f}, {y_m:.6f}, {z_m:.6f}] m -> "
            f"[{x_mm:.3f}, {y_mm:.3f}, {z_mm:.3f}] mm"
        )

        response = self._robot.ServoP(
            x_mm,
            y_mm,
            z_mm,
            float(np.rad2deg(euler[0])),
            float(np.rad2deg(euler[1])),
            float(np.rad2deg(euler[2])),
            # 1.0 / self._control_frequency,
            # self._aheadtime,
            # self._gain,
        )
        self._raise_if_dobot_error(response, "ServoP")

        return action

    def _send_gripper_action(self, action: dict[str, Any]) -> None:
        """Send gripper command using Flare Gripper.

        Action key: gripper.pos (normalized 0-1)
        """
        if not self._xense_left_gripper or not self.config.use_left_gripper or not self._xense_right_gripper or not self.config.use_right_gripper:
            self.logger.warn("No Xense Gripper connected, skipping gripper action.")
            return

        if self._left_gripper_key not in action:
            self.logger.warn("Left gripper key not in action, skipping gripper action.")
            return
        if self._right_gripper_key not in action:
            self.logger.warn("Right gripper key not in action, skipping gripper action.")
            return

        # Set gripper position
        self._xense_left_gripper.set_gripper_position(action[self._left_gripper_key])
        self._xense_right_gripper.set_gripper_position(action[self._right_gripper_key])

    def clear_fault(self) -> bool:
        """Attempt to clear robot fault.

        Returns:
            True if fault was cleared, False otherwise
        """
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.left_feedData.RobotMode != 9 or self.right_feedData.RobotMode != 9:
            self.logger.info("No fault to clear.")
            return True

        self.logger.info("Attempting to clear fault...")
        response = self._left_robot.ClearError()
        response = self._right_robot.ClearError()
        self._raise_if_dobot_error(response, "ClearError")
        self.logger.info("✅ Fault cleared successfully.")
        return True

    def disconnect(self) -> None:
        """Disconnect from the robot safely.

        This method ensures safe disconnection even if errors occur during the process.
        It will attempt to:
        1. Move robot to home position (if possible)
        2. Stop any ongoing motion
        3. Disconnect gripper and cameras
        4. Clean up all resources

        All errors are caught and logged, but the disconnect process continues to ensure cleanup.
        """
        if not self._left_is_connected or not self._right_is_connected:
            self.logger.warn(f"{self} is not connected, skipping disconnect.")
            return

        try:
            self.logger.info("Disconnecting from Dobot Nova5 robot...")

            # Move to home position before disconnecting
            try:
                self._go_to_home()
            except Exception as e:
                self.logger.warn(f"Failed to move to home before disconnect: {e}")

            # Stop any ongoing motion
            if self._left_robot is not None:
                self._left_robot.Stop()
                self._left_robot.close()
            if self._right_robot is not None:
                self._right_robot.Stop()
                self._right_robot.close()
            if self._left_feedFour is not None:
                self._left_feedFour.close()
            if self._right_feedFour is not None:
                self._right_feedFour.close()

            # Disconnect XenseFlare (provides gripper + camera + sensors)
            if self._xense_left_gripper and self.config.use_left_gripper:
                self._xense_left_gripper.disconnect()
            if self._xense_right_gripper and self.config.use_right_gripper:
                self._xense_right_gripper.disconnect()

            # Disconnect external cameras
            for cam in self.cameras.values():
                cam.disconnect()

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        finally:
            self._left_robot = None
            self._right_robot = None
            self._left_feedFour = None
            self._right_feedFour = None
            self._xense_left_gripper = None
            self._xense_right_gripper = None
            self._left_is_connected = False
            self._right_is_connected = False
            self._left_current_mode = None
            self._right_current_mode = None
            self.logger.info("✅ Dobot Nova5 disconnected.")
