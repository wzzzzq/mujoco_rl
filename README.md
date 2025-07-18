# MuJoCo RL: Piper Robot Apple Grasping

A reinforcement learning project using MuJoCo physics simulation to train a Piper robot arm to grasp an apple on a desk using visual observations from multiple cameras.

## 📋 Project Overview

This project implements a vision-based reinforcement learning system where a 7-DOF Piper robot arm learns to grasp an apple placed on a desk. The agent uses camera observations from two viewpoints (wrist camera and third-person view) to learn the grasping task through PPO (Proximal Policy Optimization).

### Key Features
- **Vision-based learning**: Uses camera observations instead of direct state information
- **Contact detection**: Sophisticated reward system based on gripper-object contact
- **Multi-camera setup**: Wrist camera and third-person camera for comprehensive visual input
- **Dynamic simulation**: Full physics simulation with MuJoCo
- **7-DOF control**: 6 rotation joints + 1 gripper joint

## 🎯 Task Description

**Goal**: Train the Piper robot arm to successfully grasp an apple placed on a desk.

**Success criteria**: 
- End effector reaches within 3cm of the apple
- Gripper makes contact with the apple
- Stable grasping behavior

**Observation space**: 
- Wrist camera: 640x480x3 RGB image
- Third-person camera: 640x480x3 RGB image

**Action space**: 7-dimensional continuous control [-1, 1] for each joint

## 🏗️ Project Structure

```
mujoco_il_rl/
├── model_assets/
│   └── piper_on_desk/
│       ├── scene.xml          # MuJoCo scene definition
│       └── assets/            # 3D models and textures
├── visual_rl/
│   └── env/
│       ├── single_piper_on_desk_env.py  # Main training environment
│       └── go2_piper_env.py             # Alternative environment
├── test.py                    # Test trained model
├── utils/
│   └── mujoco_viewer.py      # Visualization utilities
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
# Required packages
pip install numpy
pip install mujoco
pip install gymnasium
pip install stable-baselines3[extra]
pip install torch torchvision
pip install opencv-python
pip install scipy
pip install glfw
```

### Training the Model

1. **Basic training** (200K steps):
```bash
cd visual_rl/env/
python single_piper_on_desk_env.py
```

2. **Training with visualization**:
```bash
python single_piper_on_desk_env.py --render
```

3. **Custom training parameters**:
```bash
python single_piper_on_desk_env.py --n_envs 1
```

### Testing the Trained Model

After training, test the learned policy:

```bash
python test.py
```

This will load the trained model (`piper_ik_ppo_model.zip`) and demonstrate the apple-grasping behavior.

## ⚙️ Configuration

### Training Parameters

The training uses PPO with the following key parameters:

```python
# PPO Configuration
n_steps = 10           # Steps per rollout
batch_size = 50        # Mini-batch size
n_epochs = 10          # Training epochs per update
learning_rate = 3e-4   # Learning rate
gamma = 0.99           # Discount factor
episode_len = 200      # Max steps per episode
total_timesteps = 200000  # Total training steps
```

### Environment Parameters

```python
# Robot Configuration
joint_limits = [(-2.618, 2.618), (0, 3.14), ...]  # 7 joint limits
workspace_limits = {
    'x': (0.1, 0.7),
    'y': (-0.7, 0.7), 
    'z': (0.1, 0.7)
}

# Reward Components
base_reward          # Distance to apple (arctan-shaped)
contact_reward       # Bonus for gripper-apple contact (max: 5.0)
success_bonus        # Large bonus for successful grasp (20.0)
proximity_bonus      # Small bonus for getting close (5.0)
table_penalty        # Penalty for hitting table (-1.0 max)
```

## 🎮 Controls and Actions

The robot is controlled through 7 continuous actions:

1. **Joint 1**: Base rotation (-2.618 to 2.618 rad)
2. **Joint 2**: Shoulder pitch (0 to 3.14 rad) 
3. **Joint 3**: Shoulder roll (-2.697 to 0 rad)
4. **Joint 4**: Elbow pitch (-1.832 to 1.832 rad)
5. **Joint 5**: Wrist roll (-1.22 to 1.22 rad)
6. **Joint 6**: Wrist pitch (-3.14 to 3.14 rad)
7. **Gripper**: Open/close (0 to 0.035 rad)

Actions are normalized to [-1, 1] and automatically mapped to joint limits.

## 📊 Training Progress

Monitor training progress through:

### Tensorboard Logs
```bash
tensorboard --logdir ./ppo_piper/
```

### Key Metrics
- **ep_rew_mean**: Average cumulative reward per episode
- **ep_len_mean**: Average episode length (max 200)
- **success_rate**: Percentage of successful grasps
- **fps**: Training speed (steps per second)

### Expected Training Results
- **Early training** (0-50K steps): Random exploration, negative rewards
- **Learning phase** (50K-150K steps): Gradual improvement, occasional successes
- **Convergence** (150K+ steps): Consistent apple grasping, positive rewards

## 🔧 Customization

### Modify Reward Function

Edit `_compute_reward()` in `single_piper_on_desk_env.py`:

```python
def _compute_reward(self, observation):
    # Customize reward components here
    reward_components = {
        'base_reward': base_reward,
        'contact_reward': contact_reward,
        'success_bonus': success_bonus,
        # Add your custom rewards
    }
```

### Change Training Parameters

Modify the PPO configuration:

```python
model = PPO(
    "MultiInputPolicy",
    env,
    n_steps=20,           # Increase for more stable training
    batch_size=100,       # Increase for better sample efficiency
    learning_rate=1e-4,   # Decrease for more stable learning
    total_timesteps=500000  # Increase for longer training
)
```

### Add New Objects

1. Modify `scene.xml` to add new objects
2. Update `_reset_object_pose()` for object positioning
3. Modify reward function to include new objects

## 🐛 Troubleshooting

### Common Issues

1. **"Body named 'apple' not found"**
   - Ensure `scene.xml` contains the apple object
   - Check the object name matches in the code

2. **Training is very slow**
   - Reduce `total_timesteps` for faster testing
   - Use GPU acceleration: `device="cuda"`
   - Decrease image resolution if needed

3. **Robot arm goes to strange positions**
   - Check joint limits are correctly set
   - Verify action mapping in `map_action_to_joint_limits()`

4. **Camera images are black/corrupted**
   - Ensure GLFW is properly installed
   - Check camera names in `_get_image_from_camera()`

### Debug Mode

Enable detailed logging:

```python
# In single_piper_on_desk_env.py
model = PPO(..., verbose=2)  # More detailed output
```

## 📈 Performance Tips

### Training Speed
- Use GPU: `device="cuda"`
- Reduce image resolution
- Increase `n_steps` for better vectorization
- Use multiple environments: `n_envs > 1` (without rendering)

### Learning Efficiency
- Start with smaller `total_timesteps` for quick iterations
- Tune reward scaling for faster convergence
- Use curriculum learning: start with closer apple positions

### Memory Usage
- Reduce batch size if running out of memory
- Lower image resolution
- Use fewer environments

## 📄 License

This project is provided for educational and research purposes.

## 🤝 Contributing

Feel free to submit issues, feature requests, or improvements!

## 📚 References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

---

For questions or support, please open an issue in the repository.
