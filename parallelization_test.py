from gym.vector import AsyncVectorEnv
import gym
import os

# Define the environment creation function
def make_env(env_id, seed):
    def _f():
        env = gym.make(env_id)
        env.seed(seed)
        print(f"Environment {env_id} with seed {seed} is running on PID {os.getpid()}")
        return env
    return _f

num_envs = 100
env_id = 'CartPole-v1'

# Create a vector of environments
envs = AsyncVectorEnv([make_env(env_id, i) for i in range(num_envs)])

# Now you can interact with the environments
observations = envs.reset()

for i in range(1000):
    #Randomly sample actions
    actions = [envs.action_space.sample() for _ in range(num_envs)]
    observations, rewards, dones, infos = envs.step(actions)
    