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
from scipy.spatial.transform import Rotation as Rotation
from stable_baselines3.common.evaluation import evaluate_policy
import argparse
import cv2
import glfw

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm") # 忽略 stable_baselines3 中某些无害的警告，避免输出干扰

class PiperEnv(gym.Env):
    """
    Piper robot arm environment for apple grasping task using delta (incremental) actions.
    
    The neural network outputs delta joint angles [δq1, δq2, δq3, δq4, δq5, δq6, δq7] 
    representing incremental changes to joint positions rather than absolute positions.
    This approach provides smoother control and better learning stability.
    """
    def __init__(self, render=False):
        super(PiperEnv, self).__init__()
        script_dir = os.path.dirname(os.path.realpath(__file__
        )) # 获取当前脚本文件所在目录
        xml_path = os.path.join(script_dir, '../..', 'model_assets', 'piper_on_desk', 'scene.xml') # scene.xml 模型文件的完整路径，用于加载 MuJoCo 模型

        self.model = mujoco.MjModel.from_xml_path(xml_path) # 包含模型的静态信息(如几何结构、关节参数等)
        self.data = mujoco.MjData(self.model) # 保存模型的动态状态(如位置、速度、力等)
        self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6') # 末端执行器(夹爪) ID

        self.render_mode = render # 是否启用实时渲染窗口
        if self.render_mode:
            self.handle = mujoco.viewer.launch_passive(self.model, self.data) # 创建一个被动渲染窗口(GUI)，可以实时查看仿真过程
            self.handle.cam.distance = 3 # 相机与目标的距离为 3
            self.handle.cam.azimuth = 0 # 方位角为 0 度
            self.handle.cam.elevation = -30 # 仰角为 -30 度
        else:
            self.handle = None

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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,)) # 动作空间: 每个关节的角度增量 [-1,1]
        self.observation_space = spaces.Dict({
            "3rd": spaces.Box(low=0, high=255, shape=(3, 480, 640), dtype=np.uint8) 
        }) 
        self.np_random = None # 用于生成随机数，将在 reset(seed=...) 中初始化
        self.step_number = 0 # 记录当前 episode 的步数

        self.workspace_limits = {
            'x' : (0.1, 0.7),
            'y' : (-0.7, 0.7),
            'z' : (0.1, 0.7)
        } # 定义机械臂末端允许到达的空间范围

        self.goal_reached = False # 标记目标是否已达成
        self._reset_noise_scale = 0.0 # 初始状态扰动的噪声幅度，可用于增加训练多样性

        # 初始化目标 pose
        # self.goal_pos = None
        # self.goal_quat = None
        # self.goal_angle = None
        # self._set_goal_pose()

        self.episode_len = 300 # 每个 episode 的最大步数
        self.init_qpos = np.zeros(8) + 0.01 # 初始关节位置
        self.init_qvel = np.zeros(8) + 0.01 # 初始关节速度

        # 使用 GLFW 创建一个不可见的 OpenGL 窗口(用于离线渲染)
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(640, 480, "offscreen", None, None) # 用于渲染图像而不显示 GUI 界面
        glfw.make_context_current(self.window)

        self.camera = mujoco.MjvCamera() # 创建一个固定类型的相机对象，用于从指定视角渲染图像
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 
        self.scene = mujoco.MjvScene(self.model, maxgeom=1000) # MuJoCo 场景对象，用于管理渲染的几何体
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150) # MuJoCo 渲染上下文，用于实际绘图操作
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context) # 设置渲染缓冲区为离线渲染模式(不显示到屏幕)
    
    def _get_site_pos_ori(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id == -1:
            raise ValueError(f"未找到名为 '{site_name}' 的site")

        # 位置
        position = np.array(self.data.site(site_id).xpos) # site 在世界坐标系下的位置 shape (3,); t_{w}

        # 方向: MuJoCo 已存成 9 元素向量，无需 reshape
        xmat = np.array(self.data.site(site_id).xmat) # site 的旋转矩阵(按行排列); shape (9,); R_{w, site}
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, xmat) # [w, x, y, z]

        return position, quaternion
    
    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """
        通过 body 名称获取其位姿信息, 返回一个 7 维向量
            :param body_name: body 名称字符串
            :return: 7 维 numpy 数组, 格式为 [x, y, z, w, x, y, z]
            :raises ValueError: 如果找不到指定名称的body
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")
        
        # 提取位置和四元数并合并为一个 7 维向量
        position = np.array(self.data.body(body_id).xpos) # [x, y, z]
        quaternion = np.array(self.data.body(body_id).xquat) # [w, x, y, z]
        
        return position, quaternion

    # 将强化学习 agent 输出的动作值 action 从 [-1, 1] 范围映射到关节角度增量
    def map_action_to_joint_deltas(self, action: np.ndarray) -> np.ndarray:
        """
        将 [-1, 1] 范围内的 action 映射到关节角度增量 (delta angles)。
        Args:
            action (np.ndarray): 形状为 (7,) 的数组，值范围在 [-1, 1]，表示各关节的增量方向和大小
        Returns:
            np.ndarray: 形状为 (7,) 的数组，映射到实际关节角度增量，类型为 numpy.ndarray
        """
        # 定义每个关节的最大增量步长 (弧度)
        max_delta_per_step = np.array([
            0.1,   # joint 1: ±0.1 rad per step (~5.7 degrees)
            0.1,   # joint 2: ±0.1 rad per step
            0.1,   # joint 3: ±0.1 rad per step  
            0.1,   # joint 4: ±0.1 rad per step
            0.1,   # joint 5: ±0.1 rad per step
            0.1,   # joint 6: ±0.1 rad per step
            0.2,   # gripper: ±0.2 units per step
        ])
        
        # action 范围 [-1, 1] 直接映射到增量范围 [-max_delta, +max_delta]
        delta_action = action * max_delta_per_step
        
        return delta_action
    
    def apply_joint_deltas_with_limits(self, current_qpos: np.ndarray, delta_action: np.ndarray) -> np.ndarray:
        """
        将增量动作应用到当前关节位置，同时确保不超出关节限制
        Args:
            current_qpos: 当前关节位置 (7,)
            delta_action: 关节位置增量 (7,)
        Returns:
            np.ndarray: 应用增量后的新关节位置，限制在关节范围内
        """
        # 计算新的关节位置
        new_qpos = current_qpos + delta_action
        
        # 获取关节限制
        lower_bounds = self.joint_limits[:, 0]
        upper_bounds = self.joint_limits[:, 1]
        
        # 将新位置限制在关节范围内
        new_qpos = np.clip(new_qpos, lower_bounds, upper_bounds)
        
        return new_qpos
    
    # 设置 MuJoCo 模型的状态(位置 qpos 和速度 qvel)，并推进一步仿真
    def _set_state(self, qpos, qvel):
        assert qpos.shape == (8,) and qvel.shape == (8,)
        self.data.qpos[:8] = np.copy(qpos)
        self.data.qvel[:8] = np.copy(qvel)
        mujoco.mj_step(self.model, self.data) # mujoco 仿真向前推进一步，使状态生效


    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        # 定义在初始化状态中加入噪声的上下限范围
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale


        ## TODO step 1 : 把机械臂归零
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=8
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=8
        )
        
        self._set_state(qpos, qvel) # 设置模型状态并前进一步仿真使其生效

        ## TODO step 2 : 把交互的物品重新放置位置
        self._reset_object_pose()
        
        obs = self._get_observation() # 获取初始观测值

        ## TODO 以下两个保持不动
        self.step_number = 0 # 初始化 step 计数器
        self.goal_reached = False # 可用于判断是否完成任务

        return obs, {}
    
    def set_goal_pose(self, goal_body_name, position, quat_wxyz):
        # 设置 target 的位姿
        # goal_body_name = "target"
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body_name)

        if goal_body_id == -1:
            raise ValueError(f"Body named '{goal_body_name}' not found in the model.")

        # 获取 joint ID 和 qpos 起始索引
        goal_joint_id = self.model.body_jntadr[goal_body_id]
        goal_qposadr = self.model.jnt_qposadr[goal_joint_id]

        # 设置位姿
        if goal_qposadr + 7 <= self.model.nq:
            self.data.qpos[goal_qposadr: goal_qposadr + 3] = position
            self.data.qpos[goal_qposadr + 3: goal_qposadr + 7] = quat_wxyz
        else:
            print("[警告] target 的 qpos 索引越界或 joint 设置有误")
    
    def _reset_object_pose(self):
        # 将苹果随机放置在桌子中心的圆形区域内，确保相机始终可见
        item_name = "apple"
        self.target_position, item_quat = self._get_body_pose(item_name)
        
        # ----------------------------随机目标位置----------------------------
        # 桌子中心位置 (基于 scene.xml 中桌子的 pos="0 0 0.73")
        desk_center_x = 0.0
        desk_center_y = 0.0
        
        # 在桌面中心的圆形区域内随机放置，半径设为桌子宽度的一半以确保在桌面范围内
        max_radius = 0.1
        
        # 使用极坐标生成随机位置
        theta = np.random.uniform(0, 2 * np.pi)  # 完整圆形范围
        rho = max_radius * np.sqrt(np.random.uniform(0, 1))  # 均匀分布在圆内
        
        x_world_target = rho * np.cos(theta) + desk_center_x
        y_world_target = rho * np.sin(theta) + desk_center_y

        self.target_position[0] = x_world_target
        self.target_position[1] = y_world_target
        self.target_position[2] = 0.768  # Set apple on the table surface
        
        self.set_goal_pose("apple", self.target_position, item_quat)

    # 从 MuJoCo 中的某个指定名称的相机(camera)渲染图像，并返回一个 NumPy 数组形式的 OpenCV 图像(BGR 格式)
    def _get_image_from_camera(self, w, h, camera_name):
        viewport = mujoco.MjrRect(0, 0, w, h) # 定义了一个矩形区域，用于渲染图像，(0, 0, w, h) 表示从左上角 (0, 0) 开始，宽 w，高 h
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) # 使用 mj_name2id() 通过 camera_name 查找相机 ID
        self.camera.fixedcamid = cam_id # 设置 fixedcamid 表示当前要渲染的固定相机 ID
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(), 
            None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        ) # 构建渲染场景数据到 self.scene 中
        mujoco.mjr_render(viewport, self.scene, self.context) # 在指定的 viewport 上使用 self.scene 和 OpenGL 上下文 self.context 进行渲染
        rgb = np.zeros((h, w, 3), dtype=np.uint8) # 创建一个空的 RGB 图像数组，大小为 h x w x 3
        mujoco.mjr_readPixels(rgb, None, viewport, self.context) # 将渲染结果读入 rgb 数组中，第二个参数是深度图(这里传 None 表示不需要)
        cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR) # MuJoCo 渲染出来的图像上下颠倒，需要用 np.flipud 翻转回来; cv2.cvtColor(..., COLOR_RGB2BGR): 将 RGB 转换为 BGR，适配 OpenCV 默认格式
        return cv_image
    

    def _get_observation(self):
        wrist_cam_image = self._get_image_from_camera(640, 480, "wrist_cam")
        top_cam_image = self._get_image_from_camera(640, 480, "3rd")
        # 将图像从 (H, W, C) 转换为 (C, H, W)
        wrist_cam_image = np.transpose(wrist_cam_image, (2, 0, 1)) # 新形状: (3, 480, 640)
        top_cam_image = np.transpose(top_cam_image, (2, 0, 1)) # 新形状: (3, 480, 640)
        obs = {
            "3rd": top_cam_image
        }
        return obs
    

    def _check_contact_between_bodies(self, body1_name: str, body2_name: str) -> tuple[bool, float]:
        """
        检查两个 body 之间是否有接触，并返回接触力的大小
        Args:
            body1_name: 第一个body的名称
            body2_name: 第二个body的名称
        Returns:
            tuple: (是否接触, 接触力大小)
        """
        # 获取 body ID
        body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
        body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_name)
        
        if body1_id == -1 or body2_id == -1:
            return False, 0.0
            
        # 遍历所有接触点
        total_force = 0.0
        contact_found = False
        
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            # 获取接触的两个 geom
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # 获取这些 geom 所属的 body
            geom1_body = self.model.geom_bodyid[geom1_id]
            geom2_body = self.model.geom_bodyid[geom2_id]
            
            # 检查是否是我们感兴趣的两个 body 之间的接触
            if ((geom1_body == body1_id and geom2_body == body2_id) or 
                (geom1_body == body2_id and geom2_body == body1_id)):
                contact_found = True
                
                # 计算接触力大小 (法向力)
                contact_force = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, contact_force)
                force_magnitude = np.linalg.norm(contact_force[:3])  # 只考虑法向力
                total_force += force_magnitude
                
        return contact_found, total_force

    def _check_gripper_contact_with_object(self, object_name: str) -> tuple[bool, float]:
        """
        检查夹爪(包括两个手指)是否与指定物体接触
        Args:
            object_name: 物体名称
        Returns:
            tuple: (是否接触, 总接触力)
        """
        # 检查两个手指是否与物体接触
        finger1_contact, finger1_force = self._check_contact_between_bodies("link7", object_name)
        finger2_contact, finger2_force = self._check_contact_between_bodies("link8", object_name)
        
        # 任一手指接触即认为夹爪接触
        gripper_contact = finger1_contact or finger2_contact
        total_force = finger1_force + finger2_force
        
        return gripper_contact, total_force

    def _check_gripper_contact_with_table(self) -> tuple[bool, float]:
        """
        检查夹爪是否与桌面接触
        Returns:
            tuple: (是否接触, 总接触力)
        """
        # 检查夹爪各部分与桌面的接触
        for i in range(1,9,1):
            link_name = f"link{i}"
            link_contact, link_force = self._check_contact_between_bodies(link_name, "desk")
            total_force = 0.0
            contact_found = False
            if link_contact:
                # 如果任一部分接触，记录接触信息
                contact_found = True
                # print(f"夹爪与桌面接触: {link_name} 接触力 {link_force}")
                total_force += link_force

        return contact_found, total_force

    def _check_apple_fell_off_table(self) -> bool:
        """
        检查苹果是否从桌子上掉落
        Returns:
            bool: 是否掉落
        """
        apple_position, _ = self._get_body_pose('apple')
        x, y, z = apple_position
        
        # 桌子的边界 (从 scene.xml 获取: pos="0 0 0.73", size="0.3 0.6 0.01115")
        table_x_min, table_x_max = -0.3, 0.3
        table_y_min, table_y_max = -0.6, 0.6
        table_surface_height = 0.74115  # 0.73 + 0.01115
        
        # 检查是否在桌面 x,y 范围内
        if x < table_x_min or x > table_x_max:
            return True

        if y < table_y_min or y > table_y_max:
            return True

        # 检查是否掉到桌面以下 (给一些容错空间，如果苹果低于桌面5cm则认为掉落)
        if z < table_surface_height - 0.05:
            return True

        return False

    def _compute_distance_to_rest_qpos(self):
        """计算当前关节角度与初始休息位置的距离"""
        current_qpos = self.data.qpos[:7]
        rest_qpos = self.init_qpos[:7]  # 使用初始化位置作为休息位置
        return np.linalg.norm(current_qpos - rest_qpos)

    def _compute_reward(self):
        # Stage 1: Reward for reaching and grasping the apple
        # Stage 2: If grasping, reward for returning to rest pose  
        # Penalty for touching the table throughout
        
        # 获取当前末端执行器位姿和目标物体位姿
        end_ee_position, _ = self._get_site_pos_ori("end_ee")
        apple_position, _ = self._get_body_pose('apple')
        
        # 计算TCP到目标物体的距离
        tcp_to_obj_dist = np.linalg.norm(end_ee_position - apple_position)
        
        # Stage 1: 接近奖励 - 使用tanh函数提供平滑的距离奖励
        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        
        # 检查是否抓取到物体
        apple_contact, apple_force = self._check_gripper_contact_with_object('apple')
        is_grasped = 1.0 if apple_contact and apple_force > 0.5 else 0.0  # 需要一定的接触力才算抓取
        
        # 基础奖励: 接近奖励 + 抓取奖励
        reward = reaching_reward + is_grasped
        
        # Stage 2: 如果已抓取，奖励返回休息位置
        if is_grasped > 0:
            distance_to_rest = self._compute_distance_to_rest_qpos()
            place_reward = np.exp(-2 * distance_to_rest)
            reward += place_reward
            
            if tcp_to_obj_dist < 0.03 and distance_to_rest < 0.2:
                self.goal_reached = True
        
        # 惩罚接触桌面
        table_contact, table_force = self._check_gripper_contact_with_table()
        touching_table = 1.0 if table_contact else 0.0
        reward -= 2 * touching_table
        
        # 保存奖励组件用于调试和分析
        self.reward_components = {
            'reaching_reward': reaching_reward,
            'is_grasped': is_grasped,
            'place_reward': place_reward * is_grasped if is_grasped > 0 else 0.0,
            'table_penalty': -2 * touching_table,
            'tcp_to_obj_dist': tcp_to_obj_dist,
            'distance_to_rest': self._compute_distance_to_rest_qpos() if is_grasped > 0 else 0.0,
            'total_reward': reward
        }
        
        return reward
        

    # 接收一个动作 action，执行一步环境逻辑，并返回观测值、奖励、终止信号等信息
    def step(self, action):
        # 将 action 映射为关节角度增量
        delta_action = self.map_action_to_joint_deltas(action)
        
        # 获取当前关节位置
        current_qpos = self.data.qpos[:7].copy()
        
        # 应用增量并确保在关节限制内
        new_qpos = self.apply_joint_deltas_with_limits(current_qpos, delta_action)
        
        # 设置新的关节目标位置
        self.data.ctrl[:7] = new_qpos
        
        for _ in range(200):
            mujoco.mj_step(self.model, self.data) # mujoco 仿真向前推进一步，此处会做动力学积分，更新所有物理状态(位置、速度、接触力等)
            if self.handle is not None:
                self.handle.sync()
            
            current_qpos = self.data.qpos[:7].copy() # 更新当前关节位置
            pos_err = np.linalg.norm(new_qpos - current_qpos) # 计算新旧关节位置的误差
            # print(f"Step {_+1}: Position error = {pos_err:.4f}") # 打印每一步的误差
            if pos_err < 0.08:
                 break

        self.step_number += 1 # 更新步数计数器
        observation = self._get_observation() # 获取当前状态观测值

        reward = self._compute_reward() # 计算 reward

        apple_fell = self._check_apple_fell_off_table()
        done = self.goal_reached or apple_fell # 任务完成或苹果掉落则结束当前 episode

        # 创建详细的info字典，包含奖励组件信息
        info = {
            'is_success': done,
            'reward_components': self.reward_components.copy() if hasattr(self, 'reward_components') else {},
            'total_reward': reward,
            'step_number': self.step_number,
            'goal_reached': self.goal_reached,
            'current_qpos': current_qpos.copy(),  # 添加当前关节位置信息
            'delta_action': delta_action.copy(),  # 添加增量动作信息
            'new_qpos': new_qpos.copy()          # 添加新关节位置信息
        } 

        truncated = self.step_number > self.episode_len # 如果当前步数超过了预设的最大步数，则 episode 被截断，通常表示未完成任务但时间到了。
        
        return observation, reward, done, truncated, info

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        return [seed]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PiperEnv RL simulation.")
    parser.add_argument('--render', action='store_true', help='Enable rendering with GUI viewer')
    parser.add_argument('--n_envs', type=int, default=1, help='Number of envs')
    args = parser.parse_args()

    # 增加检查逻辑
    if args.render and args.n_envs != 1:
        raise ValueError("Rendering is only supported with --n_envs=1")

    # 创建 agent 交互环境
    env = make_vec_env(lambda: PiperEnv(render=args.render), n_envs=args.n_envs)

    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
    )

    """
    训练参数
    参数名	                   解释	                                建议与注意事项
    learning_rate	          学习率	                           固定值适合起步，调 schedule 可提升稳定性
    n_steps	                  每个环境每次 rollout 的步数            必须满足 n_steps * n_envs > 1, 推荐设为 128~2048
    batch_size	              每次优化的最小 batch 大小	             
    n_epochs	              每次更新重复训练的次数	              增加样本利用率， 3~10 是常用区间
    gamma	                  奖励折扣因子（长期 vs 短期）	          0.95~0.99 之间，任务长期性越强设得越高
    device	                  训练使用设备	                        GPU / CPU
    tensorboard_log           训练日志保存地址
    """
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./ppo_piper/"
    )

    """
    参数名	                  解释	                                   
    total_timesteps          总共与环境交互的步数 (env.step() 的次数总和）   
    progress_bar             是否显示训练进度条
    """
    model.learn(total_timesteps=50000000, progress_bar=True)
    model.save("piper_ik_ppo_model")

    print(" model sava success ! ")

    ### 继续训练
    # model = PPO.load("./piper_ppo_model.zip")
    # model.set_env(env)
    # model.learn(total_timesteps=2048*100, progress_bar=Truget_action_and_value)


    """
    🔁 rollout/ 部分（环境交互结果）
    ep_len_mean	           每个 episode 平均的步数 (本例为 200)
    ep_rew_mean	           每个 episode 平均的累计 reward
    success_rate	       每个 episode 是否完成成功任务的比例

    ⏱ time/ 部分（训练时间相关）
    fps	                   每秒仿真多少步
    iterations	           算法已完成的优化周期
    time_elapsed	       总共训练的时间, 单位是秒
    total_timesteps	       总共采样过的环境步数

    🎯 train/ 部分（策略学习质量
    approx_kl	           当前策略和旧策略之间的 KL 散度（衡量变化幅度）合理范围约 0.01~0.03
    clip_fraction	       有多少动作被 clip_range 限制 (PPO 的核心) 较高表示训练波动大
    clip_range	           PPO 的超参数，常见默认值为 0.2
    entropy_loss	       策略分布的熵（探索性）。越大越随机，越小越确定
    explained_variance	   Critic 的预测值和真实 return 的相关性 1 表示完全拟合，< 0 表示预测很差
    learning_rate	       当前学习率
    loss	               总损失 (值函数 + 策略 + entropy)
    n_updates	           总共优化了多少次
    policy_gradient_loss   策略网络的梯度损失 (负值是正常的)
    std	                   策略网络输出的动作标准差（表示策略不确定性）
    value_loss	           Critic 的回归损失，越低越好
    """