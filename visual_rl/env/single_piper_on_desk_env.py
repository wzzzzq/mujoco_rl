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

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,)) # 动作空间
        self.observation_space = spaces.Dict({
            "wrist": spaces.Box(low=0, high=255, shape=(3, 480, 640), dtype=np.uint8),
            "3rd": spaces.Box(low=0, high=255, shape=(3, 480, 640), dtype=np.uint8) 
        }) # 观测空间，包含腕部相机和第三视角相机
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

        self.episode_len = 200 # 每个 episode 的最大步数
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
    
    # # set random goal position for cartesian space
    # def _label_goal_pose(self, position, quat_wxyz):
    #     """
    #     设置目标位姿(位置 + 姿态)
    #     Args:
    #         position: 目标的位置，(x, y, z)，类型为 numpy.ndarray 或 list
    #         quat_wxyz: 目标的姿态，四元数 (w, x, y, z)，类型为 numpy.ndarray 或 list
    #     """
    #     ## ====== 设置 target 的位姿 ======
    #     goal_body_name = "target"
    #     goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, goal_body_name) # 使用 MuJoCo 的 mj_name2id() 函数根据名称查找 body 的 ID

    #     if goal_body_id == -1:
    #         raise ValueError(f"Body named '{goal_body_name}' not found in the model.")

    #     # 获取 joint ID 和 qpos 起始索引
    #     goal_joint_id = self.model.body_jntadr[goal_body_id] # 获取 body 对应的关节索引
    #     goal_qposadr = self.model.jnt_qposadr[goal_joint_id] # 获取该关节在 qpos 向量中的起始索引

    #     # 设置位姿，将传入的 position 和 quat_wxyz 写入到对应的 qpos 区域中，每个自由度为 7 的关节占用 7 个连续的 qpos 元素
    #     if goal_qposadr + 7 <= self.model.nq:
    #         self.data.qpos[goal_qposadr     : goal_qposadr + 3] = position
    #         self.data.qpos[goal_qposadr + 3 : goal_qposadr + 7] = quat_wxyz
    #     else:
    #         print("[警告] target 的 qpos 索引越界或 joint 设置有误")

    # 从模型中提取某个 site 的位置和姿态，在 MuJoCo 中，site 是一种虚拟标记点，可以附加在 body 上，常用于表示末端执行器、参考坐标系等
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

    # 将强化学习 agent 输出的动作值 action 从 [-1, 1] 范围映射到实际的关节角度范围
    def map_action_to_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """
        将 [-1, 1] 范围内的 action 映射到每个关节的具体角度范围。
        Args:
            action (np.ndarray): 形状为 (7,) 的数组，值范围在 [-1, 1]
        Returns:
            np.ndarray: 形状为 (7,) 的数组，映射到实际关节角度范围，类型为 numpy.ndarray
        """
        normalized = (action + 1) / 2 # 将 action 从 [-1, 1] 映射到 [0, 1]，便于线性插值
        # 从 self.joint_limits 获取每个关节的角度下限和上限
        lower_bounds = self.joint_limits[:, 0]
        upper_bounds = self.joint_limits[:, 1]
        mapped_action = lower_bounds + normalized * (upper_bounds - lower_bounds) # 线性插值得到实际的关节角度值

        return mapped_action
    
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
        # self._set_goal_pose()

        ## TODO step 2 : 把交互的物品重新放置位置
        self._reset_object_pose()
        
        # 加入扰动，使得在每次 reset 时将物体放在略微不同的位置，提高训练鲁棒性
        obs = self._get_observation() # 获取初始观测值

        ## TODO 以下两个保持不动
        self.step_number = 0 # 初始化 step 计数器
        self.goal_reached = False # 可用于判断是否完成任务

        return obs, {}
    
    def _reset_object_pose(self):
 
        apple_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'apple')
        
        # 获取苹果对应的关节信息
        apple_joint_id = self.model.body_jntadr[apple_id]
        apple_qposadr = self.model.jnt_qposadr[apple_joint_id]
        
        # 定义苹果的初始位置范围 (在桌面上的合理范围)
        base_pos = np.array([0.02, 0.27, 0.768])  # 初始位置

        self.data.qpos[apple_qposadr:apple_qposadr + 3] = base_pos # 设置苹果的初始位置
        self.data.qpos[apple_qposadr + 3:apple_qposadr + 7] = [0, 0, 0, 0]

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
            "wrist": wrist_cam_image,
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
        # link6 是末端执行器主体
        link6_contact, link6_force = self._check_contact_between_bodies("link6", "desk")
        finger1_contact, finger1_force = self._check_contact_between_bodies("link7", "desk")
        finger2_contact, finger2_force = self._check_contact_between_bodies("link8", "desk")
        
        # 任何部分接触即认为夹爪接触桌面
        table_contact = link6_contact or finger1_contact or finger2_contact
        total_force = link6_force + finger1_force + finger2_force
        
        return table_contact, total_force

    def _compute_pos_error_and_reward(self, cur_pos, goal_pos):
        # 计算位置误差(欧氏距离)
        pos_error = np.linalg.norm(cur_pos - goal_pos) # 计算当前末端执行器位置与目标位置之间的欧氏距离误差
        pos_reward = -np.arctan(pos_error) # 使用反正切函数对位置误差进行非线性变换，并取负数，这是一个密集奖励函数，距离越小, reward 越接近 0(即越好); 距离越大, reward 趋近于 -π/2 ≈ -1.57(即越差)
        return pos_reward, pos_error
    

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
            'table_penalty': 0.0,
            'contact_reward': 0.0,
            'success_bonus': 0.0,
            'proximity_bonus': 0.0,
            'position_error': pos_err
        }
        
        # 检查夹爪与桌面的接触 - 惩罚接触桌面
        table_contact, table_force = self._check_gripper_contact_with_table()
        if table_contact:
            # 惩罚与桌面接触，力越大惩罚越重
            table_penalty = -0.5*min(table_force / 10.0, 2.0)  # 负惩罚，限制最大惩罚为 -1.0
            reward_components['table_penalty'] = table_penalty
            
        # 检查夹爪与苹果的接触 - 奖励接触苹果
        apple_contact, apple_force = self._check_gripper_contact_with_object('apple')
        if apple_contact:
            # 奖励与苹果接触，接触力适中时奖励更高
            contact_reward = 5*min(apple_force / 5.0, 1.0)  # 最大奖励 5
            reward_components['contact_reward'] = contact_reward
            
            # 如果有接触且距离很近，可以考虑成功
            if pos_err < 0.03:  # 3cm 阈值更严格
                self.goal_reached = True
                reward_components['success_bonus'] = 20.0  # 接触成功的高奖励
        elif pos_err < 0.05:  # 如果没有接触但距离很近，也给予一定奖励
            reward_components['proximity_bonus'] = 5.0
        elif pos_err > 0.2:
            # 如果距离过远，给予负奖励
            reward_components['proximity_bonus'] = -2.0
        # 存储奖励组件供step函数使用
        self.reward_components = reward_components
        
        # 计算总奖励
        total_reward = sum(reward_components.values()) - reward_components['position_error']  # position_error不计入总和
        return total_reward
        

    # 接收一个动作 action，执行一步环境逻辑，并返回观测值、奖励、终止信号等信息
    def step(self, action):
        mapped_action = self.map_action_to_joint_limits(action) # 将 action 映射回真实机械臂关节空间
        # TODO: Use delta or action?
        self.data.ctrl[:7] = mapped_action[:7] # 动作
        # self._label_goal_pose(self.goal_pos, self.goal_quat)
        # TODO: One sample should last longer
        for i in range(100):
            mujoco.mj_step(self.model, self.data) # mujoco 仿真向前推进一步，此处会做动力学积分，更新所有物理状态(位置、速度、接触力等)

        self.step_number += 1 # 更新步数计数器
        observation = self._get_observation() # 获取当前状态观测值

        reward = self._compute_reward(observation) # 计算 reward
        done = self.goal_reached
        
        # 创建详细的info字典，包含奖励组件信息
        info = {
            'is_success': done,
            'reward_components': self.reward_components.copy() if hasattr(self, 'reward_components') else {},
            'total_reward': reward,
            'step_number': self.step_number,
            'goal_reached': self.goal_reached
        } 

        truncated = self.step_number > self.episode_len # 如果当前步数超过了预设的最大步数，则 episode 被截断，通常表示未完成任务但时间到了。
        
        if self.handle is not None:
            self.handle.sync()

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
        n_steps=10,
        batch_size=50,
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
    model.learn(total_timesteps=200000, progress_bar=True)  # Changed from 20M to 100K steps
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