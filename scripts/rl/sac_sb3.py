import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized HalfCheetah environment
env = make_vec_env("HalfCheetah-v4", n_envs=1)

# Initialize SAC model
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_halfcheetah_tensorboard/")

# Train the model
model.learn(total_timesteps=500000)

# Save the trained model
model.save("sac_halfcheetah")

env.close()

# Load and test the trained model
del model  # Delete model to demonstrate loading
model = SAC.load("sac_halfcheetah")

env = gym.make("HalfCheetah-v5", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
