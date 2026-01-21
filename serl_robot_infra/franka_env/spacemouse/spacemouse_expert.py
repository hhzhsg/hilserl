import multiprocessing
import numpy as np
from franka_env.spacemouse import pyspacemouse
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provides
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6  # Using lists for compatibility
        self.latest_data["buttons"] = [0, 0, 0, 0]

        # Start a process to continuously read the SpaceMouse state
        self.process = multiprocessing.Process(target=self._read_spacemouse)
        self.process.daemon = True
        self.process.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw,
                    -state[1].y, state[1].x, state[1].z,
                    -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z,
                    -state[0].roll, -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            # Update the shared state
            self.latest_data["action"] = action
            self.latest_data["buttons"] = buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        action = self.latest_data["action"]
        buttons = self.latest_data["buttons"]
        return np.array(action), buttons
    
    # def _read_spacemouse(self):
    #     while True:
    #         state = pyspacemouse.read_all()
    #         # 初始化为标准的 6 维列表
    #         action = [0.0] * 6
    #         buttons = [0, 0]

    #         if len(state) > 0:
    #             s = state[0]
    #             # 显式构造，不使用拼接，确保只有 6 位
    #             action = [
    #                 float(-s.y), float(s.x), float(s.z),
    #                 float(-s.roll), float(s.pitch), float(-s.yaw)
    #             ]
    #             buttons = list(s.buttons)

    #         # 更新共享字典
    #         self.latest_data["action"] = action
    #         self.latest_data["buttons"] = buttons

    # def get_action(self) -> Tuple[np.ndarray, list]:
    #     """强制输出严格的 7 维向量"""
    #     # 即使后台进程出错了，这里也只取前 6 位
    #     raw_action = list(self.latest_data.get("action", [0.0]*6))
    #     action_6d = raw_action[:6] 
        
    #     buttons = list(self.latest_data.get("buttons", [0, 0]))
        
    #     gripper_action = 0.0
    #     if len(buttons) >= 2:
    #         if buttons[0]: # 左键：打开
    #             gripper_action = 1.0
    #         elif buttons[1]: # 右键：闭合
    #             gripper_action = -1.0
            
    #     # 拼接成确定的 7 维
    #     action_7d = np.array(action_6d + [gripper_action], dtype=np.float32)
        
    #     return action_7d, buttons
    
    def close(self):
        # pyspacemouse.close()
        self.process.terminate()
