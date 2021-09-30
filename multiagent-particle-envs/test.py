import numpy as np
from make_env import make_env
import gym
import sys
import numpy as np
import os
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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

import numpy as np
#from maddpg import MADDPG
#from buffer import MultiAgentReplayBuffer

floor_plan_real = [[0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

agent_skills_2task = [[0,0,0],
                      [0,0,1],
                      [0,1,0]]

env = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)

env.render()
sys.exit(0)

print('number of agents', env.n)
print('observation space', env.observation_space)
print('action space', env.action_space)
print('n actions', env.action_space[0].n)

observation = env.reset()
print("observation = ",observation)

no_op = np.array([0])

action = [no_op, no_op, no_op]
print("Action = ",action)

obs_, reward, done, info = env.step(action)

print(reward)
print(done)