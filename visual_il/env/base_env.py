import sys
import os

# 添加上一级目录到模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.mujoco_viewer import BaseViewer
import mujoco,time,threading
import numpy as np
import pinocchio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import transformations as tf
from scipy.spatial.transform import Rotation
import torch
import cv2
import glfw

### 机械臂规划相关
import ikpy.chain
from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.piper import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)


class BaseEnv(BaseViewer):
    def __init__(self, cfg):
        """
        path :       XML 模型路径(MJCF)
        distance :   相机距离
        azimuth :    水平旋转角度
        elevation :  俯视角度
        """
        super().__init__(cfg.path, 3, azimuth=180, elevation=-30)
        self.path = cfg.path

        if cfg["is_have_arm"] == True:
            # 创建机械臂逆运动学解算模型 TODO 替换成自己的 ik
            self.my_chain = ikpy.chain.Chain.from_urdf_file("/home/cfy/cfy/gs_hs/model_asserts/ik_asserts/piper_n.urdf")
            # 创建机械臂规划模型
            self.model_roboplan, self.collision_model, visual_model = load_models(use_sphere_collisions=True)
            if self.collision_model is None:
                raise ValueError("collision_model is None — collision model must be loaded before proceeding.")
            
            add_self_collisions(self.model_roboplan, self.collision_model)
            add_object_collisions(self.model_roboplan, self.collision_model, visual_model, inflation_radius=0.1)
            self.target_frame = "link6"
            np.set_printoptions(precision=3)
            self.distance_padding = 0.001
            self.index = 0


        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        self.episode_len = cfg["episode_len"]
        self.is_save_record_data = cfg["is_save_record_data"]
        self.camera_names = cfg["camera_names"]
        self.data_dict = {
            'observations': {
                'images': {cam_name: [] for cam_name in self.camera_names},
                'qpos': [],
                'actions': []
            }
        }
        self.step_number = 0
        self.goal_reached_count = 0

        # 打印当前场景 joint 和 body 信息
        self.print_all_joint_info()
        self.print_all_body_info()


    def close(self):
        super().close()
        if hasattr(self, "window") and self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None


    def print_all_joint_info(self):
        """
        打印模型中所有关节的名称、ID、范围限制和当前qpos值
        """
        print("\n=== 关节信息 ===")
        print(f"{'Joint Name':<20} {'Type':<15} {'Qpos Addr':<10} {'Range':<25} {'Current Value':<15}")
        print("-" * 90)
        
        for joint_id in range(self.model.njnt):
            # 获取关节名称
            name_addr = self.model.name_jntadr[joint_id]
            joint_name = self.model.names[name_addr:].split(b'\x00')[0].decode('utf-8')
            
            # 获取关节类型
            joint_type = self.model.jnt_type[joint_id]
            type_names = {
                0: "自由关节(6DOF)",
                1: "球关节(3DOF)", 
                2: "滑动关节",
                3: "铰链关节"
            }
            type_str = type_names.get(joint_type, "未知类型")
            
            # 获取qpos地址和范围
            qpos_addr = self.model.jnt_qposadr[joint_id]
            if self.model.jnt_limited[joint_id]:
                jnt_range = f"[{self.model.jnt_range[joint_id,0]:.2f}, {self.model.jnt_range[joint_id,1]:.2f}]"
            else:
                jnt_range = "无限制"
            
            # 获取当前值
            if joint_type == 0:  # 自由关节
                current_val = self.data.qpos[qpos_addr:qpos_addr+7]
            elif joint_type == 1:  # 球关节
                current_val = self.data.qpos[qpos_addr:qpos_addr+4]
            else:  # 滑动/铰链关节
                current_val = self.data.qpos[qpos_addr]
            
            print(f"{joint_name:<20} {type_str:<15} {qpos_addr:<10} {jnt_range:<25} {str(current_val):<15}")


    def print_all_body_info(self):
        """
        打印模型中所有body的名称、ID、位置和四元数姿态信息
        """
        print("\n=== Body 信息 ===")
        print(f"{'Body Name':<25} {'Body ID':<8} {'Position':<30} {'Quaternion':<35}")
        print("-" * 100)
        
        for body_id in range(self.model.nbody):
            # 获取body名称
            name_addr = self.model.name_bodyadr[body_id]
            body_name = self.model.names[name_addr:].split(b'\x00')[0].decode('utf-8')
            
            # 获取位置和四元数
            pos = self.data.body(body_id).xpos
            quat = self.data.body(body_id).xquat
            
            print(f"{body_name:<25} {body_id:<8} {str(pos):<30} {str(quat):<35}")

    
    def _get_sensor_data(self, sensor_name: str):
        """
        通过 sensor 名称获取传感器数据
        Args:
            sensor_name     : sensor 名字
        Return:
            sensor_values   : sensor 值
        """
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found in model!")
        start_idx = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        sensor_values = self.data.sensordata[start_idx : start_idx + dim]
        return sensor_values


    def _get_body_pose(self, body_name: str) -> np.ndarray:
        """
        通过 body 名称获取其位姿信息, 返回一个7维向量
            :param body_name: body名称字符串
            :return: 7维numpy数组, 格式为 [x, y, z, w, x, y, z]
            :raises ValueError: 如果找不到指定名称的body
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"未找到名为 '{body_name}' 的body")
        
        # 提取位置和四元数并合并为一个7维向量
        position = np.array(self.data.body(body_id).xpos)  # [x, y, z]
        quaternion = np.array(self.data.body(body_id).xquat)  # [w, x, y, z]
        
        return position, quaternion


    def _get_image_from_camera(self, w, h, camera_name):
        """
        通过 camera 名称获取其相机数据
        Args:
            w               :                   期望图像宽
            h               :                   期望图像高
            camera_name     :                   相机名称
        Return:
            cv_image        :                   np.ndarray, OpenCV 格式的图像, shape 为 (h, w, 3)
        """
        viewport = mujoco.MjrRect(0, 0, w, h)
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera.id = cam_id
        mujoco.mjv_updateScene(
            self.model, self.data, mujoco.MjvOption(), 
            None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, self.context)
        cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        return cv_image


    def _get_observation(self):
        """
        获取当前环境中多个相机视角的观测图像
        Return:
            obs     :                            dict, 包含来自多个相机的图像观测, 键为相机名称, 值为对应的 np.ndarray 图像 (OpenCV 格式, shape 为 (480, 640, 3))
        """
        wrist_cam_image = self._get_image_from_camera(640, 480, "wrist")
        top_cam_image = self._get_image_from_camera(640, 480, "3rd")
        obs = {
            "wrist": wrist_cam_image,
            "3rd": top_cam_image
        }
        return obs


    def _set_original_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """
        设置模拟器中的状态（关节位置和速度），并同步 forward。
        """
        assert qpos.shape == (self.model.nq,), f"Expected qpos shape ({self.model.nq},), got {qpos.shape}"
        assert qvel.shape == (self.model.nv,), f"Expected qvel shape ({self.model.nv},), got {qvel.shape}"

        self.data.qpos[:] = np.copy(qpos)
        self.data.qvel[:] = np.copy(qvel)

        if self.model.na == 0 and hasattr(self.data, "act"):
            # 如果没有 actuator，清空 act，否则可能会引发非法内存读写
            self.data.act[:] = 0

        self.cur_episode_done = False

        mujoco.mj_forward(self.model, self.data)

    
    def calculate_target_joint6_pos_ori(
        self,
        target_get_ee_pos,
        target_ee_quat_wxyz
    ):
        """
        计算末端夹爪到关节 6 的变换
        
        Args:
            target_get_ee_pos:                     末端夹爪在世界系下的期望位置 (x, y, z)
            target_ee_quat_wxyz:                   末端夹爪在世界系下的期望姿态 (w, x, y, z)
        Return:
            link6 pos in world frame & quat (w,x,y,z) trnafered from ee pos
                {'position': (x, y, z), 'quaternion_wxyz': (w, x, y, z)}
        """

        T_link_ee = np.zeros((4,4))
      
        ee_quat_xyzw = np.roll(target_ee_quat_wxyz, -1)

        rot_ee = Rotation.from_quat(ee_quat_xyzw).as_matrix()
       

        T_link_ee[:3,:3] = Rotation.from_matrix(rot_ee).as_matrix()
        T_link_ee[:3,3] = target_get_ee_pos 

        T_from_ee_to_link6 = np.array([[0, 0, 1, -0.085], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        link6_transfered = T_link_ee @ T_from_ee_to_link6

        link6_pos_transfered = np.array(link6_transfered[:3,3])
        link6_xyzw_transfered = Rotation.from_matrix(np.array(link6_transfered[:3,:3])).as_quat()
        link6_wxyz_transfered = np.array([link6_xyzw_transfered[3],link6_xyzw_transfered[0],link6_xyzw_transfered[1],link6_xyzw_transfered[2]])
       
        return link6_pos_transfered, link6_wxyz_transfered


    def world_to_local_pose(self, world_obj_pos, world_obj_quat, local_origin_pos, local_origin_quat):
        """
        将世界坐标系下的物体位姿转换到以点a为原点的局部坐标系下的位姿
        
        参数:
            world_obj_pos:       物体在世界坐标系下的位置 [x, y, z]
            world_obj_quat:      物体在世界坐标系下的旋转四元数 [w, x, y, z]
            local_origin_pos:    局部坐标系原点a在世界坐标系下的位置 [x, y, z]
            local_origin_quat:   局部坐标系原点a在世界坐标系下的旋转四元数 [w, x, y, z]
            
        返回:
            local_obj_pos:       物体在局部坐标系下的位置 [x, y, z]
            local_obj_quat:      物体在局部坐标系下的旋转四元数 [w, x, y, z]
        """
        # 计算局部坐标系到世界坐标系的变换矩阵
        R_world_to_local = Rotation.from_quat(local_origin_quat[[1, 2, 3, 0]]).inv()  # 注意 wxyz -> xyzw
        
        # 转换位置
        local_obj_pos = R_world_to_local.apply(world_obj_pos - local_origin_pos)
        
        # 转换旋转 (四元数乘法 q_local = q_a^-1 * q_world)
        rot_obj_world = Rotation.from_quat(world_obj_quat[[1, 2, 3, 0]])
        rot_local = R_world_to_local * rot_obj_world
        local_obj_quat = rot_local.as_quat()[[3, 0, 1, 2]]  # xyzw -> wxyz
        return local_obj_pos, local_obj_quat


    def calc_arm_rrt_cubic_traj(
        self,
        cur_joints_state,
        target_joints_state
    ):
        """
        计算机械臂在给定目标关节角度下的运动轨迹
        
        参数:
            cur_joints_state:         当前机械臂的 6 关节状态
            target_joints_state:      机械臂目标 6 关节
        返回:
            path: 轨迹
        """
        q_start = cur_joints_state
        q_goal = target_joints_state

        print(f"q_start : {q_start}")

        # Search for a path
        options = RRTPlannerOptions(
            max_step_size=0.05,
            max_connection_dist=5.0,
            rrt_connect=False,
            bidirectional_rrt=True,
            rrt_star=True,
            max_rewire_dist=5.0,
            max_planning_time=20.0,
            fast_return=True,
            goal_biasing_probability=0.15,
            collision_distance_padding=0.01,
        )
        print(f"Planning a path...")
        planner = RRTPlanner(self.model_roboplan, self.collision_model, options=options)
        q_path = planner.plan(q_start, q_goal)
        if len(q_path) > 0:
            print(f"Got a path with {len(q_path)} waypoints")
        else:
            print("Failed to plan.")

        # Perform trajectory optimization.
        dt = 0.025
        options = CubicTrajectoryOptimizationOptions(
            num_waypoints=len(q_path),
            samples_per_segment=7,
            min_segment_time=0.5,
            max_segment_time=10.0,
            min_vel=-1.5,
            max_vel=1.5,
            min_accel=-0.75,
            max_accel=0.75,
            min_jerk=-1.0,
            max_jerk=1.0,
            max_planning_time=30.0,
            check_collisions=True,
            min_collision_dist=self.distance_padding,
            collision_influence_dist=0.05,
            collision_avoidance_cost_weight=0.0,
            collision_link_list=[
                "ground_plane",
                "link6",
            ],
        )
        print("Optimizing the path...")
        optimizer = CubicTrajectoryOptimization(self.model_roboplan, self.collision_model, options)
        traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)

        if traj is not None:
            print("Trajectory optimization successful")
            traj_gen = traj.generate(dt)

        if traj is not None:
            return traj_gen[1]
        else:
            return None


    def calc_arm_plan_path(
        self,
        target_pos_world,
        target_pos_ori_world,
        arm_base_pos_world,
        arm_base_ori_world,
        cur_joints_state
    ):
        """
        计算机械臂到世界坐标系下的某一点的运动轨迹
        
        参数:
            target_pos_world:         世界坐标系下的目标位置 (xyz)
            target_pos_ori_world:     世界坐标系下的目标姿态 (wxyz)
            arm_base_pos_world:       世界坐标系下的机械臂基座的位置 (xyz)
            arm_base_ori_world:       世界坐标系下的机械臂基座的姿态 (wxyz)
            cur_joints_state:         当前机械臂的 6 关节状态
        返回:
            path:                     期望轨迹
            target_joints_state:      期望轨迹的最后 6 关节位置
        """

        ## Step 1 计算 link 6 的目标位姿 世界坐标系 !!!
        target_ee_pos = target_pos_world
        target_ee_ori = target_pos_ori_world
        target_link6_pos, target_link6_wxyz = self.calculate_target_joint6_pos_ori(target_ee_pos, target_ee_ori)
        ## Step 2 计算在 arm base 坐标系下 link 6 的目标位姿
        arm_base_pos = arm_base_pos_world
        arm_base_ori = arm_base_ori_world
        local_link6_pos, local_link6_quat_wxyz = self.world_to_local_pose(target_link6_pos, target_link6_wxyz, arm_base_pos, arm_base_ori)
        ## Step 3 调用 ik 求解 TODO 换成你自己写的 ik
        # 目标抓取位置
        target_position = local_link6_pos
        # 目标抓取姿态
        quat_xyzw = [local_link6_quat_wxyz[1], local_link6_quat_wxyz[2], local_link6_quat_wxyz[3], local_link6_quat_wxyz[0]]

        # 转成 Rotation 对象
        rotation_obj = Rotation.from_quat(quat_xyzw)

        # 转为欧拉角（单位是弧度）
        target_orientation_euler = rotation_obj.as_euler('xyz', degrees=False)

        # 再转成旋转矩阵
        target_orientation = tf.euler_matrix(*target_orientation_euler)[:3, :3]
        # 计算逆运动学解
        target_joint_angles = self.my_chain.inverse_kinematics(target_position, target_orientation, "all")
        target_joints_state = target_joint_angles[1:]

        path = self.calc_arm_rrt_cubic_traj(cur_joints_state, target_joints_state)
        return path, target_joints_state


    def run_before(self):
        # step 1 : 设置机械臂、被抓物体的初始 pose
        self.init_state = self.data.qpos.copy()
        self.q_vec = None
        
        q_start = np.zeros(6)
        if q_start is None:
            raise RuntimeError(" q_start is invalid... ")

        # step 2 : 获取机械臂基座、被抓物体在世界坐标系下的 pose
        item_name = "apple"
        item_pos, item_quat = self._get_body_pose(item_name)

        arm_base_name = "base_link"
        arm_base_pos, arm_base_quat = self._get_body_pose(arm_base_name)

        # step 3 : 调用规划算法求解抓取轨迹 -> (plan_path 是一系列的路径点, target_joints_state 是终点对应的机械臂关节角)
        plan_path, target_joints_state = self.calc_arm_plan_path(item_pos, item_quat, arm_base_pos, arm_base_quat, q_start)

        self.q_vec = plan_path

        if self.q_vec is None:
            raise RuntimeError(" planning path failed... ")


    
        

    def runFunc(self):
        pass