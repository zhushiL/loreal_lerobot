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

from lerobot.robots.dobot_nova5.TCP_IP_Python_V4.dobot_api import DobotApiFeedBack,DobotApiDashboard
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.dobot_nova5.config_dobot_nova5 import ControlMode, DobotNova5Config
from lerobot.robots.dobot_nova5.xense_gripper import Gripper
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

class DobotNova5(Robot):
    """Dobot Nova5 6-DOF collaborative robot.

    This class implements the LeRobot Robot interface for the Dobot Nova5 robot,
    supporting two control modes for joint space and Cartesian space control.

    Control Modes (NRT only for Python API):
        - JOINT_MOTION: Joint motion control (maps to NRT_JOINT_MOTION)
        - CARTESIAN_MOTION: Cartesian motion control (maps to NRT_CARTESIAN_MOTION)

    Example:
        >>> from lerobot.robots.dobot_nova5.TCP_IP_Python_V4.dobot_api import DobotApiFeedBack,DobotApiDashboard
        >>> from lerobot.robots.dobot_nova5.config_dobot_nova5 import DobotNova5Config, ControlMode
        >>> # Joint motion control
        >>> config = DobotNova5Config(
        ...     robot_ip="192.168.1.1",
        ...     control_mode=ControlMode.JOINT_MOTION,
        ... )
        >>> # Cartesian motion control
        >>> config = DobotNova5Config(
        ...     robot_ip="192.168.1.1",
        ...     control_mode=ControlMode.CARTESIAN_MOTION,
        ... )
        >>> robot = DobotNova5(config)
        >>> robot.connect()
        >>> obs = robot.get_observation()
        >>> robot.send_action({"joint_1.pos": 0.0, ...})
        >>> robot.disconnect()
    """

    config_class = DobotNova5Config
    name = "dobot_nova5"

    def __init__(self, config: DobotNova5Config):
        super().__init__(config)
        self.config = config

        # Logger
        self.logger = get_logger("DobotNova5")

        # Robot interface (initialized on connect)
        # Note: Python API only supports NRT (Non-Real-Time) modes, no Scheduler needed
        self._robot: DobotApiDashboard | None = None
        self._feedInfo = []
        self.__globalLockValue = threading.Lock()

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

        self.feedData = item()  # 定义结构对象

        self._is_connected = False

        self._xense_gripper: Gripper | None = None
        if config.use_gripper:
            self._xense_gripper = Gripper(config.xense_gripper)

        # Control state - stores the current dobot_api.Mode
        self._current_mode = None

        # Home TCP pose - stored after moving to home position
        # Format: [x, y, z, qw, qx, qy, qz] (7D) - SDK format with quaternion
        self._home_tcp_pose: np.ndarray | None = None

        # Gripper key (1D) - always used
        self._gripper_key = "gripper.pos"

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
        self._joint_pos_keys = tuple(f"joint_{i}.pos" for i in range(1, JOINT_DOF + 1))

        # Joint action keys: joint_{1-6}.pos
        self._action_joint_keys = tuple(f"joint_{i}.pos" for i in range(1, JOINT_DOF + 1))

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
        self._tcp_pose_keys = (
            "tcp.x",
            "tcp.y",
            "tcp.z",
            "tcp.r1",
            "tcp.r2",
            "tcp.r3",
            "tcp.r4",
            "tcp.r5",
            "tcp.r6",
        )

        # TCP pose action keys (same as observation keys for 6D rotation)
        self._action_tcp_pose_keys = self._tcp_pose_keys

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
            features.update(dict.fromkeys(self._action_joint_keys, float))

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            # TCP pose (9D: xyz + 6D rotation)
            features.update(dict.fromkeys(self._action_tcp_pose_keys, float))

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Always include gripper (1D)
        features[self._gripper_key] = float
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
            features.update(dict.fromkeys(self._joint_pos_keys, float))

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            # TCP pose (9D: xyz + 6D rotation)
            features.update(dict.fromkeys(self._tcp_pose_keys, float))

        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Always include gripper position (1D)
        features[self._gripper_key] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Return camera/image features from Xense Gripper and external cameras."""
        features = {}

        if self._xense_gripper and self.config.use_gripper:
            features["left_tactile"] = (
                self._xense_gripper._config.rectify_size[1],
                self._xense_gripper._config.rectify_size[0],
                3,
            )
            features["right_tactile"] = (
                self._xense_gripper._config.rectify_size[1],
                self._xense_gripper._config.rectify_size[0],
                3,
            )

        # External cameras (e.g., scene cameras)
        for cam in self.cameras:
            features[cam] = (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)

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
            self._is_connected
            and self._robot is not None
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

    def parseResultId(self, valueRecv):
        # 解析返回值，确保机器人在 TCP 控制模式
        if "Not Tcp" in valueRecv:
            print("Control Mode Is Not Tcp")
            return [1]
        return [int(num) for num in re.findall(r'-?\d+', valueRecv)] or [2]

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
            self.logger.info(f"Connecting to Dobot Nova5 robot: {self.config.robot_ip}")

            # Create robot interface
            self._robot = DobotApiDashboard(self.config.robot_ip, self.config.dashboardPort)
            self._feedFour = DobotApiFeedBack(self.config.robot_ip, self.config.feedPortFour)

            # Enable the robot
            self.logger.info("Enabling robot...")
            if self.config.control_mode == ControlMode.CARTESIAN_MOTION:
                if self.parseResultId(self._robot.EnableRobot())[0] != 0:
                    self.logger.error("Failed to enable robot. Check if port 29999 is occupied.")
                    return
                self.logger.info("Robot TCP enabled successfully.")
            elif self.config.control_mode == ControlMode.JOINT_MOTION:
                self._robot.EnableRobot()
                self.logger.info("Robot Joint enabled successfully.")
            else:
                raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

            # Start feedback thread to continuously read robot state (for get_observation)
            # TODO: Needs testing
            feed_thread = threading.Thread(
                target=self.get_robot_state)  # robot state feedback thread
            feed_thread.daemon = True
            feed_thread.start()

            # Clear any existing fault
            # TODO: RobotMode 支持的端口是29999还是30004？需要确认一下
            if self.feedData.RobotMode == 9: # ROBOT_MODE_ERROR
                self.logger.warn("Fault occurred on the connected robot, trying to clear ...")
                # Try to clear the fault
                if not self._robot.ClearError():
                    raise RuntimeError("Failed to clear robot fault. Check the robot status.")
                self.logger.info("Fault on the connected robot is cleared")
            elif self.feedData.RobotMode == 5: # ROBOT_MODE_ENABLE
                self.logger.info("Robot is enabled and ready.")
            else:
                self.logger.warn(f"Robot is in unexpected mode {self.feedData.RobotMode} after enabling. Check robot status.")

            # Wait for robot to become operational
            timeout = 30  # seconds
            start_time = time.time()
            while not self.feedData.RobotMode == 5:  # ROBOT_MODE_ENABLE  Enabled and idle
                if time.time() - start_time > timeout:
                    raise RuntimeError(f"Robot did not become operational within {timeout} seconds")
                time.sleep(0.1)

            self.logger.info("Robot is now operational.")

            # Connect Xense Gripper end-effector (provides gripper + wrist_cam + tactile)
            if self._xense_gripper and self.config.use_gripper:
                self.logger.info("Connecting Xense Gripper...")
                self._xense_gripper.connect()

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
            if self._xense_gripper and self.config.use_gripper:
                gripper_status = "with Xense Gripper (gripper + wrist_cam + tactile)"
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

        home_point_list = np.deg2rad(self.config.home_point_list)
        # 走点指令
        recvmovemess = self._robot.MovJ(*home_point_list, 0)
        print("MovJ:", recvmovemess)
        print(self.parseResultId(recvmovemess))
        currentCommandID = self.parseResultId(recvmovemess)[1]
        print("指令 ID:", currentCommandID)
        #sleep(0.02)
        while True:  #完成判断循环

            print(self.feedData.RobotMode)
            print("robotCurrentCommandID",self.feedData.robotCurrentCommandID)
            print("currentCommandID",currentCommandID)
            if self.feedData.RobotMode == 5 and self.feedData.robotCurrentCommandID == currentCommandID:
                print("运动结束")
                break
            time.sleep(0.1)
        
        # Initialize gripper position (only if gripper is enabled)
        if self._xense_gripper is not None:
            if self.config.xense_gripper_init_open:
                self._xense_gripper._gripper.set_position_sync(
                    self.config.gripper_max_pos,
                    vmax=self.config.gripper_velocity,
                    fmax=self.config.gripper_force,
                )  # fully open
            else:
                self._xense_gripper._gripper.set_position_sync(
                    self.config.gripper_min_pos,
                    vmax=self.config.gripper_velocity,
                    fmax=self.config.gripper_force
                )  # fully closed
        
        # Wait for target reached
        while not self._robot.primitive_states()["reachedTarget"]:
            time.sleep(0.1)
        self._home_tcp_pose = np.array(self._robot.states().tcp_pose)
        self.logger.info(f"Home TCP pose: {self._home_tcp_pose}")
        self.logger.info("✅ Robot at home position.")
        if self._xense_gripper is not None:
            self.logger.info(f"Gripper position: {self._xense_gripper.get_gripper_position()}")

    def _go_to_start(self) -> None:
        """Move robot to start position using MoveJ primitive.

        Uses ExecutePrimitive("MoveJ") with configurable parameters:
        - jntVelScale: Joint velocity scale 1-100 (from config.start_vel_scale)
        - target: Start joint position in degrees (from config.start_position_degree)
        """
        if not self._is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.logger.info("Moving to start position...")

        # Create target joint position from config (in degrees)
        start_jpos = self.config.start_position_degree

        # 走点指令
        recvmovemess = self._robot.MovJ(*start_jpos, 0)
        print("MovJ:", recvmovemess)
        print(self.parseResultId(recvmovemess))
        currentCommandID = self.parseResultId(recvmovemess)[1]
        print("指令 ID:", currentCommandID)
        #sleep(0.02)
        while True:  #完成判断循环

            print(self.feedData.RobotMode)
            if self.feedData.RobotMode == 5 and self.feedData.robotCurrentCommandID == currentCommandID:
                print("运动结束")
                break
            time.sleep(0.1)
        
        # Initialize gripper position (only if gripper is enabled)
        if self._xense_gripper is not None:
            if self.config.xense_gripper_init_open:
                self._xense_gripper._gripper.set_position_sync(
                    self.config.gripper_max_pos,
                    vmax=self.config.gripper_velocity,
                    fmax=self.config.gripper_force,
                )  # fully open
            else:
                self._xense_gripper._gripper.set_position_sync(
                    self.config.gripper_min_pos,
                    vmax=self.config.gripper_velocity,
                    fmax=self.config.gripper_force
                )  # fully closed
        
        # Wait for target reached
        while not self._robot.primitive_states()["reachedTarget"]:
            time.sleep(0.1)

        self.logger.info("✅ Robot at start position.")
        if self._xense_gripper is not None:
            self.logger.info(f"Gripper position: {self._xense_gripper.get_gripper_position()}")

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
            feedInfo = self._feedFour.feedBackData()
            with self.__globalLockValue:
                if feedInfo is not None:   
                    if hex((feedInfo['TestValue'][0])) == '0x123456789abcdef':
                        # 基础字段
                        self.feedData.MessageSize = feedInfo['len'][0]
                        self.feedData.RobotMode = feedInfo['RobotMode'][0]
                        self.feedData.DigitalInputs = feedInfo['DigitalInputs'][0]
                        self.feedData.DigitalOutputs = feedInfo['DigitalOutputs'][0]
                        self.feedData.robotCurrentCommandID = feedInfo['CurrentCommandId'][0]
                        self.feedData.tcpPose = feedInfo['ToolVectorActual'][0]
                        self.feedData.qActual = feedInfo['QActual'][0]
                        # 自定义添加所需反馈数据
                        '''
                        self.feedData.DigitalOutputs = int(feedInfo['DigitalOutputs'][0])
                        self.feedData.RobotMode = int(feedInfo['RobotMode'][0])
                        self.feedData.TimeStamp = int(feedInfo['TimeStamp'][0])
                        '''


    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation based on control_mode.

        Returns a dictionary with observation data. The content depends on control_mode:
        - JOINT_MOTION: joint_1-7.{pos,vel,effort} (21D) + gripper.pos (1D) = 22D
        - CARTESIAN_MOTION: tcp.{x,y,z,r1-r6} (9D) + gripper (1D) = 10D

        Also includes camera images if configured.
        """
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        if self.config.control_mode == ControlMode.JOINT_MOTION:
            # Joint positions (7D)
            for i, key in enumerate(self._joint_pos_keys):
                # TODO
                obs_dict[key] = self.feedData.qActual[i]

        elif self.config.control_mode == ControlMode.CARTESIAN_MOTION:
            # TCP pose from SDK: [x, y, z, qw, qx, qy, qz]
            # TODO
            tcp_pose = self.feedData.tcpPose

            # Position (3D)
            obs_dict["tcp.x"] = tcp_pose[0]
            obs_dict["tcp.y"] = tcp_pose[1]
            obs_dict["tcp.z"] = tcp_pose[2]

            # Convert quaternion to 6D rotation representation
            quat = euler_to_quaternion(np.deg2rad(tcp_pose[3]), np.deg2rad(tcp_pose[4]), np.deg2rad(tcp_pose[5]))
            r6d = quaternion_to_rotation_6d(quat[0], quat[1], quat[2], quat[3])

            obs_dict["tcp.r1"] = r6d[0]
            obs_dict["tcp.r2"] = r6d[1]
            obs_dict["tcp.r3"] = r6d[2]
            obs_dict["tcp.r4"] = r6d[3]
            obs_dict["tcp.r5"] = r6d[4]
            obs_dict["tcp.r6"] = r6d[5]


        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        # Get data from Xense Gripper (gripper + tactile sensors)
        if self._xense_gripper is not None and self.config.use_gripper:
            # Read sensors (keys are mapped from SN to sensor_keys names)
            sensor_data = self._xense_gripper.get_sensor_data()
            for key, data in sensor_data.items():
                obs_dict[key] = data

            # Read gripper position
            obs_dict[self._gripper_key] = self._xense_gripper.get_gripper_position()

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
        
        # Compute forward kinematics
        tcp_pose = self._robot.PositiveKin(joint_positions_deg)

        
        return np.array(tcp_pose, dtype=np.float32)

    def get_start_tcp_pose(self) -> np.ndarray:
        """Get TCP pose at start position (from config.start_position_degree).
        
        This can be called after Robot object is created, even before robot is enabled.
        
        Returns:
            TCP pose [x, y, z, qw, qx, qy, qz] (7D)
        """
        return self.forward_kinematics(self.config.start_position_degree)

    def get_start_tcp_pose_euler(self) -> np.ndarray:
        """Get TCP pose at start position in Euler format [x, y, z, roll, pitch, yaw, gripper_pos].
        
        This can be called after Robot object is created, even before robot is enabled.
        
        Returns:
            numpy array of shape (7,) with [x, y, z, roll, pitch, yaw, gripper_pos]
        """
        tcp_pose_quat = self.get_start_tcp_pose()
        
        # Convert quaternion to Euler angles
        euler = quaternion_to_euler(
            tcp_pose_quat[3], tcp_pose_quat[4], tcp_pose_quat[5], tcp_pose_quat[6]
        )
        
        # Get initial gripper position based on config
        gripper_pos = self.config.gripper_max_pos if self.config.xense_gripper_init_open else 0.0
        
        return np.array(
            [tcp_pose_quat[0], tcp_pose_quat[1], tcp_pose_quat[2], 
             euler[0], euler[1], euler[2], gripper_pos],
            dtype=np.float32,
        )

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

        # TODO
        # Get current TCP pose (quaternion format)
        states = self._robot.states()
        tcp_pose = states.tcp_pose  # [x, y, z, qw, qx, qy, qz]

        # Convert quaternion to Euler angles
        euler = quaternion_to_euler(tcp_pose[3], tcp_pose[4], tcp_pose[5], tcp_pose[6])
        roll, pitch, yaw = euler[0], euler[1], euler[2]

        # Get gripper position directly from XenseFlare gripper
        gripper_pos = 0.0
        if self._xense_gripper and self.config.use_gripper:
            gripper_pos = self._xense_gripper.get_gripper_position()

        # Return [x, y, z, roll, pitch, yaw, gripper_pos]
        return np.array(
            [tcp_pose[0], tcp_pose[1], tcp_pose[2], roll, pitch, yaw, gripper_pos],
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

        # TODO
        # Get current TCP pose (quaternion format)
        states = self._robot.states()
        tcp_pose = states.tcp_pose  # [x, y, z, qw, qx, qy, qz]

        # Get gripper position directly from XenseFlare gripper
        gripper_pos = 0.0
        if self._xense_gripper and self.config.use_gripper:
            gripper_pos = self._xense_gripper.get_gripper_position()

        # Return [x, y, z, qw, qx, qy, qz, gripper_pos]
        return np.array(
            [*tcp_pose, gripper_pos],
            dtype=np.float32,
        )

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
        if not self.is_connected or self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Check for fault
        if self._robot.fault():
            raise RuntimeError("Robot fault detected. Call robot.clear_fault() or reconnect.")

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
        self._robot.ServoJ(
            target_pos[0],
            target_pos[1],
            target_pos[2],
            target_pos[3],
            target_pos[4],
            target_pos[5],
            self._control_frequency,
            self._aheadtime,
            self._gain,
        )

        return action

    def _send_cartesian_pure_motion_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send Cartesian pure motion command (NRT mode, no force control).

        Action keys: action.tcp.{x,y,z,r1,r2,r3,r4,r5,r6}

        The action uses 6D rotation representation which is converted to quaternion
        for the Flexiv SDK (SendCartesianMotionForce expects [x,y,z,qw,qx,qy,qz]).
        """
        # Extract position
        x, y, z = action["tcp.x"], action["tcp.y"], action["tcp.z"]

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
        print("x,y,z",x,y,z)
        print("euler",euler)
        # # Send command using NRT API (pure motion - no wrench parameter needed)
        # self._robot.ServoP(
        #     x, y, z, 
        #     euler[0], euler[1], euler[2], 
        #     self._control_frequency, 
        #     self._aheadtime, 
        #     self._gain
        # )

        return action

    def _send_gripper_action(self, action: dict[str, Any]) -> None:
        """Send gripper command using Flare Gripper.

        Action key: gripper.pos (normalized 0-1)
        """
        if not self._xense_gripper or not self.config.use_gripper:
            return

        if self._gripper_key not in action:
            return

        # Set gripper position
        self._xense_gripper.set_gripper_position(action[self._gripper_key])  # normalized [0, 1]

    def clear_fault(self) -> bool:
        """Attempt to clear robot fault.

        Returns:
            True if fault was cleared, False otherwise
        """
        if self._robot is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # TODO
        if not self._robot.fault():
            self.logger.info("No fault to clear.")
            return True

        # TODO
        self.logger.info("Attempting to clear fault...")
        result = self._robot.ClearFault()
        if result:
            self.logger.info("✅ Fault cleared successfully.")
        else:
            self.logger.error("Failed to clear fault.")
        return result

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
        if not self._is_connected:
            self.logger.warn(f"{self} is not connected, skipping disconnect.")
            return

        try:
            self.logger.info("Disconnecting from Flexiv robot...")

            # Move to home position before disconnecting
            try:
                self._go_to_home()
            except Exception as e:
                self.logger.warn(f"Failed to move to home before disconnect: {e}")

            # Stop any ongoing motion
            if self._robot is not None:
                self._robot.Stop()

            # Disconnect XenseFlare (provides gripper + camera + sensors)
            if self._xense_gripper and self.config.use_gripper:
                self._xense_gripper.disconnect()

            # Disconnect external cameras
            for cam in self.cameras.values():
                cam.disconnect()

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
        finally:
            self._robot = None
            self._xense_gripper = None
            self._is_connected = False
            self._current_mode = None
            self.logger.info("✅ Flexiv Rizon4 disconnected.")
