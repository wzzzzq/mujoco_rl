
import numpy as np
import mujoco
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch.nn as nn
import warnings
import torch
import mujoco.viewer
import os
import time
from scipy.spatial.transform import Rotation as Rotation
import argparse
import cv2
import glfw

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PiperEnv(gym.Env):
    def __init__(self, render=True):
        super(PiperEnv, self).__init__()
        # 获取当前脚本文件所在目录
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # 构造 scene.xml 的完整路径
        xml_path = os.path.join(script_dir, 'model_assets', 'piper_on_desk', 'scene.xml')
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        self.render_mode = render
        if self.render_mode:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data)
            self.handle.cam.distance = 3
            self.handle.cam.azimuth = 0
            self.handle.cam.elevation = -30
        else:
            self.handle = None

        self.rl_model = PPO.load("./piper_ik_ppo_model.zip")

        # 各关节运动限位，一共 7 个自由度，前 6 个是旋转关节，最后一个是夹爪开合
        self.joint_limits = np.array([
            (-2.618, 2.618),
            (0, 3.14),
            (-2.697, 0),
            (-1.832, 1.832),
            (-1.22, 1.22),
            (-3.14, 3.14),
            (0, 0.035),
        ])

        # 动作空间，7个关节（6个旋转关节 + 1个夹爪）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,))
        # 观测空间，包含腕部相机和第三视角相机
        self.observation_space = spaces.Dict({
            "wrist": spaces.Box(low=0, high=255, shape=(3, 480, 640), dtype=np.uint8),
            "3rd": spaces.Box(low=0, high=255, shape=(3, 480, 640), dtype=np.uint8) 
        })
        self.np_random = None   
        self.step_number = 0

        #workspace limit of robot
        self.workspace_limits = {
            'x' : (0.1, 0.7),
            'y' : (-0.7, 0.7),
            'z' : (0.1, 0.7)
        }

        self.goal_reached = False
        self._reset_noise_scale = 0.0

        self.episode_len = 200  # Match the original environment
        self.init_qpos = np.zeros(8) + 0.01  # 8 DOF for robot + apple
        self.init_qvel = np.zeros(8) + 0.01

        # 使用 GLFW 创建一个不可见的 OpenGL 窗口(用于离线渲染)
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(640, 480, "offscreen", None, None)
        glfw.make_context_current(self.window)

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
    
    def _get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        通过 body 名称获取其位姿信息
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")
        
        # 提取位置和四元数
        position = np.array(self.data.body(body_id).xpos)  # [x, y, z]
        quaternion = np.array(self.data.body(body_id).xquat)  # [w, x, y, z]
        
        return position, quaternion

    def _reset_object_pose(self):
        apple_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'apple')
        
        # 获取苹果对应的关节信息
        apple_joint_id = self.model.body_jntadr[apple_id]
        apple_qposadr = self.model.jnt_qposadr[apple_joint_id]
        
        # 定义苹果的初始位置范围 (在桌面上的合理范围)
        base_pos = np.array([0.02, 0.27, 0.768])  # 初始位置

        self.data.qpos[apple_qposadr:apple_qposadr + 3] = base_pos
        self.data.qpos[apple_qposadr + 3:apple_qposadr + 7] = [1, 0, 0, 0]  # 四元数

    # 从 MuJoCo 中的某个指定名称的相机(camera)渲染图像
    def _get_image_from_camera(self, w, h, camera_name):
        viewport = mujoco.MjrRect(0, 0, w, h)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera.fixedcamid = cam_id
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(), 
            None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        return cv_image

    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 '{site_name}' 的site")

        # 位置
        position = np.array(self.data.site(site_id).xpos)        # shape (3,)

        # 方向：MuJoCo 已存成9元素向量，无需reshape
        xmat = np.array(self.data.site(site_id).xmat)            # shape (9,)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat)                    # [w, x, y, z]

        return position, quaternion

    def map_action_to_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """
        将 [-1, 1] 范围内的 action 映射到每个关节的具体角度范围。
        Args:
            action (np.ndarray): 形状为 (7,) 的数组，值范围在 [-1, 1]
        Returns:
            np.ndarray: 形状为 (7,) 的数组，映射到实际关节角度范围
        """
        normalized = (action + 1) / 2
        lower_bounds = self.joint_limits[:, 0]
        upper_bounds = self.joint_limits[:, 1]
        mapped_action = lower_bounds + normalized * (upper_bounds - lower_bounds)
        return mapped_action
    
    def _set_state(self, qpos, qvel):
        assert qpos.shape == (8,) and qvel.shape == (8,)
        self.data.qpos[:8] = np.copy(qpos)
        self.data.qvel[:8] = np.copy(qvel)
        mujoco.mj_step(self.model, self.data)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=8
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=8
        )
        
        self._set_state(qpos, qvel)
        self._reset_object_pose()
        
        obs = self._get_observation()
        self.step_number = 0
        self.goal_reached = False

        return obs, {}
    
    def _get_observation(self):
        wrist_cam_image = self._get_image_from_camera(640, 480, "wrist_cam")
        top_cam_image = self._get_image_from_camera(640, 480, "3rd")
        # 将图像从 (H, W, C) 转换为 (C, H, W)
        wrist_cam_image = np.transpose(wrist_cam_image, (2, 0, 1))
        top_cam_image = np.transpose(top_cam_image, (2, 0, 1))
        obs = {
            "wrist": wrist_cam_image,
            "3rd": top_cam_image
        }
        return obs
    
    def _compute_pos_error_and_reward(self, cur_pos, goal_pos):
        # 计算位置误差(欧氏距离)
        pos_error = np.linalg.norm(cur_pos - goal_pos)
        pos_reward = -np.arctan(pos_error)
        return pos_reward, pos_error

    def _check_contact_between_bodies(self, body1_name: str, body2_name: str) -> tuple[bool, float]:
        """检查两个 body 之间是否有接触，并返回接触力的大小"""
        body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
        
        if body1_id == -1 or body2_id == -1:
            return False, 0.0
            
        total_force = 0.0
        contact_found = False
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            geom1_body = self.model.geom_bodyid[geom1_id]
            geom2_body = self.model.geom_bodyid[geom2_id]
            
            if ((geom1_body == body1_id and geom2_body == body2_id) or 
                (geom1_body == body2_id and geom2_body == body1_id)):
                contact_found = True
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
                force_magnitude = np.linalg.norm(contact_force[:3])
                total_force += force_magnitude
                
        return contact_found, total_force

    def _check_gripper_contact_with_object(self, object_name: str) -> tuple[bool, float]:
        """检查夹爪是否与指定物体接触"""
        finger1_contact, finger1_force = self._check_contact_between_bodies("link7", object_name)
        finger2_contact, finger2_force = self._check_contact_between_bodies("link8", object_name)
        gripper_contact = finger1_contact or finger2_contact
        total_force = finger1_force + finger2_force
        return gripper_contact, total_force

    def _compute_reward(self, observation):
        # 获取当前末端执行器位姿
        end_ee_position, _ = self._get_site_pos_ori("end_ee")
        # 获取目标物体位姿
        apple_position, _ = self._get_body_pose('apple')
        # 末端执行器位姿和目标物体位姿的差异作为奖励
        base_reward, pos_err = self._compute_pos_error_and_reward(end_ee_position, apple_position)
        
        # 初始化奖励组件
        reward_components = {
            'base_reward': base_reward,
            'contact_reward': 0.0,
            'success_bonus': 0.0,
            'proximity_bonus': 0.0,
        }
        
        # 检查夹爪与苹果的接触 - 奖励接触苹果
        apple_contact, apple_force = self._check_gripper_contact_with_object('apple')
        if apple_contact:
            contact_reward = 5 * min(apple_force / 5.0, 1.0)  # 最大奖励 5
            reward_components['contact_reward'] = contact_reward
            
            if pos_err < 0.03:  # 3cm 阈值
                self.goal_reached = True
                reward_components['success_bonus'] = 20.0
        elif pos_err < 0.05:  # 如果没有接触但距离很近，也给予一定奖励
            reward_components['proximity_bonus'] = 5.0
        
        total_reward = sum(reward_components.values())
        return total_reward
    def step(self, action):
        mapped_action = self.map_action_to_joint_limits(action)
        self.data.ctrl[:7] = mapped_action[:7]  # 7 DOF control
        
        # Run physics simulation for multiple steps
        for i in range(100):
            mujoco.mj_step(self.model, self.data)

        self.step_number += 1
        observation = self._get_observation()
        
        reward = self._compute_reward(observation)
        done = self.goal_reached
        info = {'is_success': done}

        truncated = self.step_number > self.episode_len
        if self.handle is not None:
            self.handle.sync()

        return observation, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        if self.handle is not None:
            self.handle.close()
        if hasattr(self, 'window'):
            glfw.destroy_window(self.window)
            glfw.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PiperEnv RL simulation.")
    env = PiperEnv()
    observation, _ = env.reset()

    try:
        # Run the simulation loop
        while True:
            action, _states = env.rl_model.predict(observation)
            observation, reward, done, truncated, info = env.step(action)

            print("*****************************")
            print(f"reward: {reward}")
            # Get current end effector and apple positions for comparison
            end_ee_pos, _ = env._get_site_pos_ori("end_ee")
            apple_pos, _ = env._get_body_pose('apple')
            print(f"End effector pos: {end_ee_pos}")
            print(f"Apple pos: {apple_pos}")
            print(f"Distance: {np.linalg.norm(end_ee_pos - apple_pos):.4f}m")

            if done or truncated:
                if done:
                    print("Goal reached! Apple grabbed successfully!")
                    
                observation, _ = env.reset()

            time.sleep(3)  # Slower for better visualization
    finally:
        env.close()