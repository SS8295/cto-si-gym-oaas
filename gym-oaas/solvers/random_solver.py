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
from max_flow import FindMaximumMatching
from max_flow import make_circuit_video

# import wandb

floor_plan_real = [[0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

agent_skills_2task = [[1,1,0],
                      [0,1,1],
                      [1,1,1],
                      [1,1,1],
                      [1,1,1]
                      ]

single_agent = [[0,1,0]]

# wandb.init(project="MaxFlow Reward Tracking")
env = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)

print(env.possible_schedules)
print(env.agent_task_distances)




sys.exit(0)
