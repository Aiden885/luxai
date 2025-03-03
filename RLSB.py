from pathlib import Path

import pybullet_envs_gymnasium

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

# Alternatively, you can use the MuJoCo equivalent "HalfCheetah-v4"
vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
# Automatically normalize the input features and reward
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO("MlpPolicy", vec_env)
model.learn(total_timesteps=2000)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = Path("/tmp/")
model.save(log_dir / "ppo_halfcheetah")
stats_path = log_dir / "vec_normalize.pkl"
vec_env.save(stats_path)

# To demonstrate loading
del model, vec_env

# Load the saved statistics
vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
vec_env = VecNormalize.load(stats_path, vec_env)
#  do not update them at test time
vec_env.training = False
# reward normalization is not needed at test time
vec_env.norm_reward = False

# Load the agent
model = PPO.load(log_dir / "ppo_halfcheetah", env=vec_env)