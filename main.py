import gym
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

# Ensure TensorFlow does not allocate all GPU memory (useful if using a GPU)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Check the environment to ensure it is compatible with stable-baselines3
check_env(env)

# Define the policy model using Proximal Policy Optimization (PPO)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    tensorboard_log="./ppo_cartpole_tensorboard/"
)

# Set up checkpoints and evaluation callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=1000, 
    save_path='./models/',
    name_prefix='ppo_cartpole'
)

eval_callback = EvalCallback(
    env,
    best_model_save_path='./models/',
    log_path='./logs/',
    eval_freq=500,
    deterministic=True,
    render=False
)

# Train the model with callbacks for checkpointing and evaluation
model.learn(
    total_timesteps=10000, 
    callback=[checkpoint_callback, eval_callback]
)

# Save the final model
model.save("ppo_cartpole_final")

# Evaluate the trained model and visualize its performance
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Close the environment
env.close()

# Example of loading and using the saved model
loaded_model = PPO.load("ppo_cartpole_final")
obs = env.reset()
for _ in range(1000):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()

# Output tensorboard logs for monitoring (if TensorBoard is installed and running)
# To visualize the TensorBoard logs, run in the terminal:
# tensorboard --logdir=./ppo_cartpole_tensorboard/
