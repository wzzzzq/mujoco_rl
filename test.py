
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
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_rl.env.single_piper_on_desk_env import PiperEnv as BasePiperEnv

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3.common.on_policy_algorithm")

class PiperTestEnv(BasePiperEnv):
    """
    Test environment that inherits from the training environment
    Only adds the trained model loading and test-specific functionality
    """
    def __init__(self, render=True, model_path="./piper_ik_ppo_model.zip"):
        # Initialize the base environment with rendering enabled
        super().__init__(render=render)
        
        # Load the trained model
        self.rl_model = PPO.load(model_path)
        
        # Override some settings for testing
        self.episode_len = 500  # Longer episodes for better observation
        
    def predict_action(self, observation):
        """Get action from the trained model"""
        action, _states = self.rl_model.predict(observation)
        return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PiperEnv RL simulation.")
    parser.add_argument("--model_path", type=str, default="./piper_ik_ppo_model.zip", 
                        help="Path to the trained PPO model")
    args = parser.parse_args()
    
    # Create the test environment
    env = PiperTestEnv(render=True, model_path=args.model_path)
    observation, _ = env.reset()

    try:
        # Run the simulation loop
        while True:
            action = env.predict_action(observation)
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
                else:
                    print("Episode ended (truncated)")
                    
                observation, _ = env.reset()

            time.sleep(0.1)  # Slower for better visualization
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        env.close()