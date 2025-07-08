import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
import geometry_msgs.msg
from scipy.spatial.transform import Rotation as R
import copy
# import pygpg
from mujoco import MjModel, MjData, mjtObj
from termcolor import cprint
import threading
from base_env import BaseEnv
from easydict import EasyDict



class SingleArmEnv(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.path = cfg.path
        self.action = None


    def unnormalizar_gripper(self,gripper_action ):
        return gripper_action * 0.035


    def run_func(self):
        if self.q_vec is None:
            raise ValueError(f" planning path does not exist... ")
        
        self.data.ctrl[:6] = self.q_vec[:6, self.index]
        self.index += 1
        if self.index >= self.q_vec.shape[1] - 1:
            self.cur_episode_done = True



def main():
    cfg = EasyDict({
        "path": "/home/cfy/cfy/sss/model_assets/piper_on_desk/scene.xml",
        "is_have_arm": True,
        "episode_len": 100,
        "is_save_record_data": True,
        "camera_names": ["3rd_camera", "wrist_cam"],
        "env_name": "SingleArmEnv",
        "obj_list": ["desk","apple","banana"]
    })
    env = SingleArmEnv(cfg)

    # 记录数据
    for i in range(cfg["episode_len"]):
        env.run_loop()
    
    

if __name__ == "__main__":
    main()