import os
import jax
import numpy as np
import jax.numpy as jnp

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig
from experiments.usb_pickup_insertion.wrapper import USBEnv, GripperPenaltyWrapper


class EnvConfig(DefaultEnvConfig):
    SERVER_URL: str = "http://192.168.2.222:5000/"
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "148522071365",
            "dim": (640, 480),
            "exposure": 10500,
        },
        "wrist_2": {
            "serial_number": "152122076640",
            "dim": (640, 480),
            "exposure": 10500,
        },
        # 新增全局相机
        "global": {
            "serial_number": "152122079499",
            "dim": (640, 480),
            "exposure": 10500, 
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[139:459, 111:506],
        "wrist_2": lambda img: img,
        "global": lambda img: img[90:441, 72:543],
    }
    TARGET_POSE = np.array(
        [
            0.407108,
            -0.081856,
            0.206891,
            3.1099675,
            0.0146619,
            -0.0078615,
        ]
    )
    RESET_POSE = TARGET_POSE + np.array([-0.0, -0.0, 0.0, 0.0, 0.3, -0.0])
    ACTION_SCALE = np.array([0.15, 0.15, 1])
    RANDOM_RESET = False
    DISPLAY_IMAGE = True
    RANDOM_XY_RANGE = 0.1
    RANDOM_Z_RANGE = 0.1
    RANDOM_RXY_RANGE = np.pi / 12
    RANDOM_RZ_RANGE = np.pi / 6
    XY_BORDER = 0.6
    Z_BORDER_LOW = 0.6
    Z_BORDER_HIGH = 0.6

    # # 定义下界 (Low Limit)
    # ABS_POSE_LIMIT_LOW = np.array(
    #     [
    #         TARGET_POSE[0] - XY_BORDER,    # X轴最小能到哪
    #         TARGET_POSE[1] - XY_BORDER,    # Y轴最小能到哪
    #         TARGET_POSE[2] - Z_BORDER_LOW, # Z轴最低能到哪 (别太低，小心撞桌子)
    #         TARGET_POSE[3] - RANDOM_RXY_RANGE,          # 角度限制 (弧度)
    #         TARGET_POSE[4] - RANDOM_RXY_RANGE,
    #         TARGET_POSE[5] - RANDOM_RZ_RANGE,
    #     ]
    # )
    # # 定义上界 (High Limit)
    # ABS_POSE_LIMIT_HIGH = np.array(
    #     [
    #         TARGET_POSE[0] + XY_BORDER,    # X轴最大能到哪
    #         TARGET_POSE[1] + XY_BORDER,    # Y轴最大能到哪
    #         TARGET_POSE[2] + Z_BORDER_HIGH,# Z轴最高能到哪
    #         TARGET_POSE[3] + RANDOM_RXY_RANGE,
    #         TARGET_POSE[4] + RANDOM_RXY_RANGE,
    #         TARGET_POSE[5] + RANDOM_RZ_RANGE,
    #     ]
    # )
    
    # 定义下界 (Low Limit)
    ABS_POSE_LIMIT_LOW = np.array(
        [
            0.4,
            -0.3,
            0.03,
            TARGET_POSE[3] - RANDOM_RXY_RANGE,          # 角度限制 (弧度)
            TARGET_POSE[4] - RANDOM_RXY_RANGE,
            TARGET_POSE[5] - RANDOM_RZ_RANGE,
        ]
    )
    # 定义上界 (High Limit)
    ABS_POSE_LIMIT_HIGH = np.array(
        [
            0.7,
            0.2,
            0.3,
            TARGET_POSE[3] + RANDOM_RXY_RANGE,
            TARGET_POSE[4] + RANDOM_RXY_RANGE,
            TARGET_POSE[5] + RANDOM_RZ_RANGE,
        ]
    )
    
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.005,
        "translational_clip_y": 0.005,
        "translational_clip_z": 0.005,
        "translational_clip_neg_x": 0.005,
        "translational_clip_neg_y": 0.005,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.02,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.02,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 3000,
        "translational_damping": 89,
        "rotational_stiffness": 300,
        "rotational_damping": 9,
        "translational_Ki": 0.1,
        "translational_clip_x": 0.01,
        "translational_clip_y": 0.01,
        "translational_clip_z": 0.01,
        "translational_clip_neg_x": 0.01,
        "translational_clip_neg_y": 0.01,
        "translational_clip_neg_z": 0.01,
        "rotational_clip_x": 0.05,
        "rotational_clip_y": 0.05,
        "rotational_clip_z": 0.05,
        "rotational_clip_neg_x": 0.05,
        "rotational_clip_neg_y": 0.05,
        "rotational_clip_neg_z": 0.05,
        "rotational_Ki": 0.1,
    }
    MAX_EPISODE_LENGTH = 150


class TrainConfig(DefaultTrainingConfig):
    image_keys = ("wrist_1", "wrist_2", "global")
    classifier_keys = ["wrist_1", "wrist_2", "global"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    checkpoint_period = 2000
    cta_ratio = 2
    random_steps = 0
    discount = 0.98
    buffer_period = 1000
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-learned-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = USBEnv(
            fake_env=fake_env, save_video=save_video, config=EnvConfig()
        )
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        if classifier:
            classifier = load_classifier_func(
                key=jax.random.PRNGKey(0),
                sample=env.observation_space.sample(),
                image_keys=self.classifier_keys,
            checkpoint_path="/home/user/hzh/hil-serl/examples/experiments/usb_pickup_insertion/classifier_ckpt",
            )

            # def reward_func(obs):
            #     sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
            #     return int(sigmoid(classifier(obs)) > 0.7 and obs["state"][0, 0] > 0.4)
            def reward_func(obs):
                sigmoid = lambda x: 1 / (1 + jnp.exp(-x))
                prob = sigmoid(classifier(obs))
                gripper_state = obs["state"][0, 0]
                
                # gripper > 0.05 表示夹爪张开（USB 已插入并松手）
                is_success = jnp.logical_and(prob > 0.9, gripper_state > 0.075)
                print('prob:', prob, 'gripper_state', gripper_state)
                return int(is_success.item())
            
            env = MultiCameraBinaryRewardClassifierWrapper(env, reward_func)
        env = GripperPenaltyWrapper(env, penalty=-0.02)
        return env