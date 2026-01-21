from typing import OrderedDict
from franka_env.camera.rs_capture import RSCapture
from franka_env.camera.video_capture import VideoCapture
from franka_env.utils.rotations import euler_2_quat
import numpy as np
import requests
import copy
import gymnasium as gym
import time
from franka_env.envs.franka_env import FrankaEnv

class USBEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def init_cameras(self, name_serial_dict=None):
        """Init both wrist cameras."""
        if self.cap is not None:
            self.close_cameras()

        self.cap = OrderedDict()
        for cam_name, kwargs in name_serial_dict.items():
            cap = VideoCapture(
                RSCapture(name=cam_name, **kwargs)
            )
            self.cap[cam_name] = cap

    def reset(self, **kwargs):
        """
        Reset 流程：
        1. 打开夹爪
        2. 移动到安全高度
        3. 移动到初始位置（娃娃上方）
        4. 等待开始
        """
        self._recover()
        self._update_currpos()
        
        # 打开夹爪
        self._send_gripper_command(1.0)
        time.sleep(0.3)
        
        # 移动到安全高度
        self._update_currpos()
        safe_pose = copy.deepcopy(self.currpos)
        safe_pose[2] = self.config.RESET_POSE[2]  # 使用 RESET_POSE 的高度
        self.interpolate_move(safe_pose, timeout=0.5)
        
        # 移动到初始位置
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.interpolate_move(self.config.RESET_POSE, timeout=0.8)
        time.sleep(0.3)
        
        # 获取初始观测
        obs, info = super().reset(**kwargs)
        return obs, info
    
    def interpolate_move(self, goal: np.ndarray, timeout: float):
        """Move the robot to the goal position with linear interpolation."""
        if goal.shape == (6,):
            goal = np.concatenate([goal[:3], euler_2_quat(goal[3:])])
        self._send_pos_command(goal)
        time.sleep(timeout)
        self._update_currpos()
    
    def go_to_reset(self, joint_reset=False):
        """自定义 reset 流程"""
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # 打开夹爪
        self._send_gripper_command(0.08)
        time.sleep(0.3)
        
        # 移动到 reset 位置
        reset_pose = self.resetpos.copy()
        self.interpolate_move(reset_pose, timeout=1)

        # 切换到柔顺模式
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.02):
        super().__init__(env)
        assert env.action_space.shape == (7,), f"Expected 7D action, got {env.action_space.shape}"
        self.penalty = penalty
        self.last_gripper_pos = None
        self.gripper_threshold = 0.04

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = obs["state"][0, 0]
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        if "intervene_action" in info:
            action = info["intervene_action"]

        current_gripper_pos = observation["state"][0, 0]
        
        # 修复：只在夹爪状态真正切换时惩罚
        last_open = self.last_gripper_pos > self.gripper_threshold
        current_open = current_gripper_pos > self.gripper_threshold
        
        if last_open != current_open:
            info["grasp_penalty"] = self.penalty
        else:
            info["grasp_penalty"] = 0.0

        self.last_gripper_pos = current_gripper_pos
        return observation, reward, terminated, truncated, info
