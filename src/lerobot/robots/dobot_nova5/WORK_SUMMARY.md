# Dobot Nova5 接入阶段性工作总结

在 `robots` 目录下新增了 Dobot Nova5 机器人接入代码，当前阶段以官方文档为主要依据完成接口设计与代码骨架搭建。由于暂时没有真机，部分接口处于占位或待验证状态。

## 已完成的工作

- 已经把最难的“接口边界定义、控制模式抽象、数据结构对齐、外设抽象”打通。
- 后续接入真机时，主要工作会从“架构设计”转为“字段对齐 + 时序调试 + 稳定性验证”，整体风险和改动范围都更可控。

### 1) 新建并组织了 Dobot Nova5 相关模块
- `src/lerobot/robots/dobot_nova5/dobot_nova5.py`
- `src/lerobot/robots/dobot_nova5/config_dobot_nova5.py`
- `src/lerobot/robots/dobot_nova5/xense_gripper.py`
- `src/lerobot/robots/dobot_nova5/config_xense_gripper.py`
- `src/lerobot/robots/dobot_nova5/__init__.py`
- `src/lerobot/robots/dobot_nova5/TCP_IP_Python_V4/*`（官方 TCP/IP Python 示例与接口文件）

### 2) 完成了机器人配置层设计
- 注册了 `RobotConfig` 子类：`dobot_nova5`。
- 定义了控制模式枚举：关节控制和笛卡尔控制。
- 暴露了核心参数：
  - 机器人通信参数（IP、端口、控制频率）
  - 起始位姿参数（`start_position_degree`、`start_vel_scale`）
  - 夹爪参数（开合范围、速度、力度、开机初始化行为）
  - 触觉传感器映射参数（SN 到特征名）
  - 相机参数入口（`cameras`）
- 在 `__post_init__` 中加入了关键参数校验逻辑（长度范围、取值区间等）。

### 3) 完成了机器人主类骨架与核心流程
- 基于 `Robot` 抽象类实现了 `DobotNova5`。
- 完成了动作/观测特征定义逻辑（随控制模式切换）。
- 完成了连接主流程框架：
  - 建立 Dashboard/Feedback 接口
  - 使能机器人
  - 启动反馈线程
  - 故障检查与清理流程骨架
  - 可选回零/回起始位姿
  - 外设（夹爪、相机）连接
- 实现了主要控制接口框架：
  - `send_action`
  - `_send_joint_position_action`
  - `_send_cartesian_pure_motion_action`
  - `_send_gripper_action`
  - `disconnect`
- 实现了与姿态表示相关的转换路径（四元数、欧拉角、6D 旋转表示）。

### 4) 完成了 Xense 夹爪集成封装
- 实现了夹爪连接/断开流程。
- 实现了夹爪位置读写（归一化到 `[0, 1]`）。
- 实现了触觉传感器扫描、连接与数据读取。
- 支持将传感器 SN 映射为训练/记录使用的语义键名。

## 当前状态
- 代码已经完成了“可扩展骨架 + 主要接口定义 + 外设集成框架”。
- 设计方向清晰，和 LeRobot 的抽象接口保持一致。
- 当前实现属于“文档驱动开发阶段”，距离“真机稳定运行阶段”还差最后一轮接口实测和联调。

## 目前待补齐或待真机验证的部分
- `get_observation` 中的部分反馈字段映射仍是占位逻辑，需要根据真实反馈包补齐。
- 部分故障处理与状态判定逻辑需要实机确认（尤其是机器人模式码与端口行为）。
- 部分函数中仍有 `TODO`，需要在真机联调时落地具体实现。
- 当前文件存在语法层面的占位残留，`py_compile` 会在 `dobot_nova5.py` 的观测分支处报错（注释占位导致代码块不完整）。
