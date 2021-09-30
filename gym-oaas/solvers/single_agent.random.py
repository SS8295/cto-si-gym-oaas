import gym
import sys
import numpy as np
import random
import time
from IPython.display import clear_output
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from gym.envs.registration import register
from os.path import isfile, join
import yaml
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from lib.dqn import *
from lib.utils import *

import ray
from ray import tune
from ray.tune.registry import register_env

style.use("ggplot")

#random.seed(1)

floor_plan_real = [[0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

agent_skills_2task = [[1,1,1]
                      ]

single_agent = [[0,1,0]]



def env_creator(env_config):
    environment = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)
    return environment  # return an env instance



#ray.init()
#register_env("stelios", env_creator)
#config = {"framework": "torch", "env": "stelios"}
#tune.run("PPO", name="stelios", config=config)
#sys.exit(0)


print("Creating Environment...")
env = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)
print("Environment Created!")

episode_reward = []
done = False
obs = env.reset()
#print(env.agent_skills)
#sys.exit(0)
for i in range(1000):
    #env.render()
    action = env.action_space.sample()
    #print(action)
    #sys.exit(0)
    obs, reward, done, info = env.step(action)
    #print(action_to_schedule(action))
    episode_reward.append(reward)
    #env.render()

#env.close()
print("Sum of all rewards = ", sum(episode_reward))
plt.plot(episode_reward)
plt.show()

# Observation should be dictionary instead of matrix
# Reward should be dictionary
