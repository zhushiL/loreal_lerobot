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

"""DH Robotics AG-95 gripper driver via the Dobot arm's built-in RS485 end-effector port.

Communication is abstracted through ModbusRTUProtocol so this class has no dependency
on DobotApiDashboard. The caller (BiDobotNova5) is responsible for establishing the
Modbus RTU connection and passing it in via connect().

Modbus RTU register map
-----------------------
Write (FC=0x06 → SetHoldRegs):
    0x0100  Initialization  — write 0xA5 (165) to trigger hardware self-init
    0x0101  Target force    — 20–100 (%)
    0x0103  Target position — 0 (closed) … 1000 (open)
    0x0104  Target speed    — 0–100 (%) (this parameter does not take effect)

Read (FC=0x03 → GetHoldRegs):
    0x0200  Init state      — 0 = not init, 1 = init done
    0x0201  Grip state      — 0 = in motion, 1 = reached, 2 = gripped, 3 = dropped
    0x0202  Current position — 0 (closed) … 1000 (open)

Position convention
-------------------
DH hardware : 0 = fully closed, 1000 = fully open
lerobot norm: 0.0 = fully closed, 1.0 = fully open

Conversion:
    dh_pos     = int(round(normalized * 1000))
    normalized = dh_pos / 1000.0

Threading note
--------------
All Modbus calls go through DobotApiDashboard's TCP socket (port 29999), shared with
arm control commands (ServoJ / ServoP). DobotApiDashboard.sendRecvMsg is protected by
an internal __globalLock, so concurrent calls from the background poll thread and the
main control thread are automatically serialized — no additional locking is needed.

A daemon thread polls _REG_CUR_POS at ~20 Hz and updates _cached_position.
get_gripper_position() returns this cached value (non-blocking).
"""

from __future__ import annotations

import time
from threading import Thread
from typing import Protocol

from lerobot.robots.bi_dobot_nova5_dh.config_dh_gripper_integrated import (
    DHGripperIntegratedConfig,
)
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.utils.robot_utils import get_logger

# ── Modbus RTU registers ───────────────────────────────────────────────────────
_REG_INITIALIZE = 0x0100
_REG_FORCE = 0x0101
_REG_POSITION = 0x0103
_REG_SPEED = 0x0104
_REG_INIT_STATE = 0x0200
_REG_GRIP_STATE = 0x0201
_REG_CUR_POS = 0x0202

_INIT_TRIGGER = 0xA5  # value to write to _REG_INITIALIZE to start self-init
_INIT_DONE = 1  # value returned by _REG_INIT_STATE when init is complete

# DH hardware position limits
_DH_POS_CLOSED = 0
_DH_POS_OPEN = 1000

# Grip state register values
_GRIP_IN_MOTION = 0
_GRIP_REACHED = 1  # target reached without contact
_GRIP_GRIPPED = 2  # gripped an object
_GRIP_DROPPED = 3  # object dropped

_POLL_INTERVAL_S          = 0.05    # 20 Hz background position poll
_POLL_FAIL_LOG_THRESHOLD  = 20
_POLL_FAIL_REPEAT_INTERVAL = 100
_SYNC_POLL_INTERVAL_S     = 0.05


class ModbusRTUProtocol(Protocol):
    """Interface expected by DHGripperIntegrated for Modbus RTU register access.

    BiDobotNova5 provides a concrete implementation (_DobotModbusRTU) that forwards
    calls through DobotApiDashboard. Any object satisfying this Protocol is accepted.
    """

    def read_register(self, reg: int) -> int | None:
        """Read a single holding register. Returns the value or None on failure."""
        ...

    def write_register(self, reg: int, value: int) -> bool:
        """Write a single holding register. Returns True on success."""
        ...

    def close(self) -> None:
        """Release the Modbus RTU master connection."""
        ...


class DHGripperIntegrated:
    """DH Robotics AG-95 gripper controlled via the Dobot arm's RS485 end-effector port.

    Requires a ModbusRTUProtocol instance to be passed to connect(). BiDobotNova5
    creates this via its internal _DobotModbusRTU class before calling connect().

    Normalized position convention:
        0.0  →  fully closed  (DH register value = 0)
        1.0  →  fully open    (DH register value = 1000)

    Example::

        cfg = DHGripperIntegratedConfig(slave_id=1, baudrate=115200)
        g = DHGripperIntegrated(cfg, name="right")
        modbus = _DobotModbusRTU(robot_dashboard, cfg.slave_id, cfg.baudrate)
        g.connect(modbus)
        g.set_gripper_position(0.5)
        print(g.get_gripper_position())
        g.disconnect()
    """

    config_class = DHGripperIntegratedConfig

    def __init__(self, config: DHGripperIntegratedConfig, name: str = "gripper"):
        self._config = config
        self._logger = get_logger(f"DHGripperIntegrated-{name}")
        self._is_connected: bool = False

        self._modbus: ModbusRTUProtocol | None = None

        # Pre-seeded from init_open; updated by the background poller.
        self._cached_position: float = 1.0 if config.init_open else 0.0

        self._poll_thread: Thread | None = None
        self._poll_running: bool = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_register(self, reg: int, value: int) -> bool:
        if self._modbus is None:
            return False
        return self._modbus.write_register(reg, value)

    def _read_register(self, reg: int) -> int | None:
        if self._modbus is None:
            return None
        return self._modbus.read_register(reg)

    # ── Background position poller ────────────────────────────────────────────

    def _position_poll_loop(self) -> None:
        """Daemon thread: continuously refresh _cached_position from hardware.

        DobotApiDashboard.sendRecvMsg is protected by an internal __globalLock,
        so concurrent calls from this thread and the main control thread are
        automatically serialized — no additional locking is needed here.
        """
        consecutive_failures = 0
        while self._poll_running:
            raw = self._read_register(_REG_CUR_POS)
            if raw is None:
                consecutive_failures += 1
                if consecutive_failures == _POLL_FAIL_LOG_THRESHOLD:
                    self._logger.error(
                        f"DHGripperIntegrated: position cache stale after "
                        f"{consecutive_failures} consecutive read failures."
                    )
                elif (
                    consecutive_failures > _POLL_FAIL_LOG_THRESHOLD
                    and (consecutive_failures - _POLL_FAIL_LOG_THRESHOLD)
                    % _POLL_FAIL_REPEAT_INTERVAL
                    == 0
                ):
                    self._logger.error(
                        f"DHGripperIntegrated: still unhealthy — "
                        f"{consecutive_failures} consecutive failures."
                    )
                time.sleep(_POLL_INTERVAL_S)
                continue

            if consecutive_failures >= _POLL_FAIL_LOG_THRESHOLD:
                self._logger.info(
                    f"DHGripperIntegrated: communication recovered after "
                    f"{consecutive_failures} consecutive failures."
                )
            consecutive_failures = 0

            raw_clamped = max(_DH_POS_CLOSED, min(raw, _DH_POS_OPEN))
            self._cached_position = raw_clamped / _DH_POS_OPEN
            time.sleep(_POLL_INTERVAL_S)

    # ── Hardware initialisation ───────────────────────────────────────────────

    def _hardware_initialize(self) -> None:
        """Trigger and wait for the DH gripper hardware self-initialisation.

        Sends 0xA5 to register 0x0100. Polls register 0x0200 until the gripper
        reports init-done (value = 1) or init_timeout is exceeded.

        Raises:
            RuntimeError: If the gripper does not finish initialising in time.
        """
        init_state = self._read_register(_REG_INIT_STATE)
        if init_state == _INIT_DONE:
            self._logger.info("DH gripper already initialised.")
            return

        self._logger.info("Triggering DH gripper hardware initialisation...")
        if not self._write_register(_REG_INITIALIZE, _INIT_TRIGGER):
            raise RuntimeError(
                "DHGripperIntegrated: failed to send initialisation command."
            )

        deadline = time.monotonic() + self._config.init_timeout
        while time.monotonic() < deadline:
            state = self._read_register(_REG_INIT_STATE)
            if state == _INIT_DONE:
                self._logger.info("DH gripper hardware initialisation complete.")
                return
            time.sleep(0.1)

        raise RuntimeError(
            f"DHGripperIntegrated: hardware initialisation timed out "
            f"after {self._config.init_timeout:.1f} s."
        )

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self, modbus: ModbusRTUProtocol) -> None:
        """Initialise the gripper using an already-established Modbus RTU connection.

        The caller (BiDobotNova5) is responsible for creating the Modbus RTU master
        (via ModbusRTUCreate) before calling this method.

        Args:
            modbus: A ModbusRTUProtocol instance providing read/write/close.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            RuntimeError: If hardware initialisation fails.
        """
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self._modbus = modbus
        self._is_connected = True
        self._cached_position = 1.0 if self._config.init_open else 0.0

        try:
            self._hardware_initialize()
        except Exception as exc:
            self._logger.error(
                f"DHGripperIntegrated hardware initialisation failed: {exc}"
            )
            try:
                modbus.close()
            except Exception:
                pass
            self._modbus = None
            self._is_connected = False
            raise

        self._write_register(_REG_FORCE, self._config.gripper_force)

        if self._config.init_open:
            self._logger.info("Opening gripper to initial position...")
            self._write_register(_REG_POSITION, _DH_POS_OPEN)
            self._cached_position = 1.0

        self._poll_running = True
        self._poll_thread = Thread(target=self._position_poll_loop, daemon=True)
        self._poll_thread.start()

        self._logger.info("DH gripper connected via robot RS485 Modbus RTU.")

    def disconnect(self) -> None:
        """Open the gripper and close the Modbus RTU master.

        Raises:
            DeviceNotConnectedError: If not connected.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._poll_running = False
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None

        self._logger.info("Opening DH gripper before disconnect...")
        self._write_register(_REG_POSITION, _DH_POS_OPEN)

        if self._modbus is not None:
            self._modbus.close()
            self._modbus = None

        self._is_connected = False
        self._logger.info("DH gripper disconnected.")

    # ── Position interface ────────────────────────────────────────────────────

    def get_gripper_position(self) -> float:
        """Return the current normalized gripper position in [0, 1].

        Returns the value cached by the background poller (non-blocking).
        Returns 0.0 if not connected.

        Returns:
            0.0 = fully closed, 1.0 = fully open.
        """
        if not self._is_connected:
            return 0.0
        return self._cached_position

    def set_gripper_position(self, normalized_pos: float) -> None:
        """Send a position command (non-blocking).

        Args:
            normalized_pos: Target in [0, 1]. 0.0 = closed, 1.0 = open.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("DH gripper is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")
        dh_pos = int(round(normalized_pos * _DH_POS_OPEN))
        if self._write_register(_REG_POSITION, dh_pos):
            self._cached_position = normalized_pos

    def set_gripper_position_sync(
        self,
        normalized_pos: float,
        timeout: float = 10.0,
    ) -> None:
        """Send a position command and block until the gripper stops moving.

        Polls the grip-state register until the gripper reports target-reached (1),
        gripped-object (2), or object-dropped (3), or until timeout elapses.

        Args:
            normalized_pos: Target in [0, 1]. 0.0 = closed, 1.0 = open.
            timeout:        Max wait time in seconds (default 10.0).
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("DH gripper is not connected.")
        if not 0.0 <= normalized_pos <= 1.0:
            raise ValueError(f"normalized_pos must be in [0, 1], got {normalized_pos}.")

        dh_pos = int(round(normalized_pos * _DH_POS_OPEN))
        if self._write_register(_REG_POSITION, dh_pos):
            self._cached_position = normalized_pos

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            grip_state = self._read_register(_REG_GRIP_STATE)
            if grip_state is not None and grip_state != _GRIP_IN_MOTION:
                return
            time.sleep(_SYNC_POLL_INTERVAL_S)

        self._logger.warning(
            f"set_gripper_position_sync timed out after {timeout:.1f} s "
            f"(target normalized={normalized_pos:.3f}, dh_pos={dh_pos})."
        )

    def initialize_gripper_position(self, normalized_pos: float = 1.0) -> None:
        """Move gripper to normalized_pos using a blocking sync command.

        Intended for use during robot start-up (before the teleoperation loop).

        Args:
            normalized_pos: Target in [0, 1]. Defaults to 1.0 (fully open).
        """
        self.set_gripper_position_sync(normalized_pos)
