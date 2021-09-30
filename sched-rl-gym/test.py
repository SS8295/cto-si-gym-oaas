import gym
import schedgym.envs as schedgym

env = gym.make('DeepRM-v0', use_raw_state=True)
env.reset()

for _ in range(200):
  env.render()
  observation, reward, done, info = env.step(env.action_space.sample())
env.close()