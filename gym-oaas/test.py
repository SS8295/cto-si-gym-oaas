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

from max_flow import FindMaximumMatching
from max_flow import make_circuit_video

#from stable_baselines3 import PPO
#from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.evaluation import evaluate_policy
#from stable_baselines3.common.env_util import make_vec_env
#from stable_baselines3.common.env_checker import check_env


# with open('floor_plans.yaml') as f:
#     floor_plan_real = yaml.safe_load(f)

# with open('agent_skills.yaml') as f:
#     agent_skills_1task = yaml.safe_load(f)

floor_plan_almost_empty = [[0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,1,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

floor_plan3_almost_full = [[1,1,1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1,1,1],
               [1,1,1,1,1,1,1,1,1,0]]

floor_plan4_long = [[0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

agent_skills_1task = [[0],
                [0],
                [1],
                [1]]

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
                      [0,1,0],
                      [0,0,1],
                      ]

env = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)

print(len(env.employee_dict))
print(len(env.task_dict))

#sys.exit(0)

# -------- Max Flow Scheduler - Baseline ------- #
for iter in range(5):

    #env.fetch_new_tasks()

    print("Reached this and printing data")
    print(env.employee_dict)
    print(env.task_dict)

    max_flow_schedule = np.zeros((len(env.agent_skills),len(env.agent_skills[0])))
    vertices = []
    edges = []

    for key in env.employee_dict:
        vertices.append(key)
    for key in env.task_dict:
        vertices.append(key)

    for agent in env.employee_dict:
        for skill in range(len(env.employee_dict[agent].skill)):
            print("Agent = ", agent)
            print("Skill = ", skill)
            print(env.employee_dict[agent].skill[skill])
            if env.employee_dict[agent].skill[skill] == 1:
                edges.append({agent,env.employee_dict[agent].task_dict_map[skill]})

    print("Vertices: ",vertices)
    print("Edges: ",edges)

    f = FindMaximumMatching(edges, vertices)
    f.find_maximum_matching()
    print("Schedule complete: ",f.matching)

    empl_idx = []
    task_idx = []

    for matching in f.matching:
        for node in matching:
            if 'employee' in node:
                idx = node.replace('employee', '')
                empl_idx.append(int(idx))
            else:
                task_idx.append(env.task_dict_inverse_map[node])

    max_flow_schedule.astype(int)
    max_flow_schedule = max_flow_schedule.tolist()

    #print("Employee indices = ", empl_idx)
    #print("Task indices = ", task_idx)
    #print("max_flow_sched_size = (", len(max_flow_schedule),",",len(max_flow_schedule[0]),")")
    #print(len(empl_idx))
    for i in range(len(empl_idx)):
        #print("At iteration ", i, "setting maxflow to 1: (",empl_idx[i],",",task_idx[i],")")
        #print(i)
        max_flow_schedule[empl_idx[i]][task_idx[i]] = 1

    
    print(max_flow_schedule)

    task_action = [[1,0,0],
                [0,0,1],
                [0,1,0],
                [0,0,1],
                [0,1,0]]

    #print(type(task_action))
    #print(type(max_flow_schedule))


            
    #sys.exit(0)
    #make_circuit_video('animation.gif', fps=1)

    #sys.exit(0)

    # -------- High Level Scheduler ------- #

    # env.render()

    # sys.exit(0)
    # task_action = [[1,0,0],
    #                [0,0,1],
    #                [0,1,0],
    #                [0,0,1],
    #                [0,1,0]]

    for i in range(15):
        env.schedule_step(max_flow_schedule)
        env.render()
        #env.create_tasks

make_circuit_video('animation.gif', fps=1)
sys.exit(0)
# -------- Reward Function Demonstration ------- #

# for i in range(500):
#     #print("i = ",i)
#     #print("i % 2  ",i %2)
#     if i<25:
#         if i % 2 == 0:
#             action = []
#             for i in range(len(env.action_space)):
#                 action_i = 1
#                 action.append(action_i)

#         elif i % 2 == 1:
#             action = []
#             for i in range(len(env.action_space)):
#                 action_i = 2
#                 action.append(action_i)

#     elif i>=25 and i < 60:
#         if i % 2 == 0:
#             action = []
#             for i in range(len(env.action_space)):
#                 action_i = 3
#                 action.append(action_i)

#         elif i % 2 == 1:
#             action = []
#             for i in range(len(env.action_space)):
#                 action_i = 4
#                 action.append(action_i)

#     elif i>=60:
#         if i % 2 == 0:
#             action = []
#             for i in range(len(env.action_space)):
#                 action_i = 1
#                 action.append(action_i)

#         elif i % 2 == 1:
#             action = []
#             for i in range(len(env.action_space)):
#                 action_i = 1
#                 action.append(action_i)

#     env.step(action)
#     env.render()

# sys.exit(0)

# -------- Random Action Behavior ------- #

for i in range(15):
    action = []
    for i in range(len(env.action_space)):
        action_i = env.action_space[i].sample()
        action.append(action_i)
    env.step(action)
    env.render()

env.close()
sys.exit(0)


# -------- MADDPG information ------- #

print('number of agents', env.n)
print('observation space', env.observation_space)
print('action space', env.action_space)
print('n actions', env.action_space[0].n)

observation = env.reset()
print(observation)

#no_op = 0
no_op = np.array([1,0,0,0,0])

action = [no_op]
print("Action = ",action)

obs_, reward, done, info = env.step(action)


print(reward)
print(done)
env.render()

sys.exit(0)

# -------- Random Action Behavior ------- #

for i in range(5):
    action = []
    for i in range(len(env.action_space)):
        action_i = env.action_space[i].sample()
        action.append(action_i)
    env.step(action)
    env.render()

#env.close()
sys.exit(0)

# -------- Archive ------- #

print("Action = ",action)

obs_, reward, done, info = env.step(action)
print(reward)
print(done)
sys.exit(0)
print(env.reset())


sys.exit(0)

for i in range(10):
    action = []
    for i in range(len(env.action_space)):
        action_i = env.action_space[i].sample()
        action.append(action_i)
    env.step(action)
    env.render()

sys.exit(0)

env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)
model.learn(total_timesteps=100000)
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
model.save(PPO_path)

sys.exit(0)
log_path = os.path.join('Training', 'Logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=100000)
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
model.save(PPO_path)


sys.exit(0)


#env = make_vec_env('gym_oaas:oaas-v0', floor_plan=floor_plan1, agent_skills=agent_skills1, n_envs=4)
env = DummyVecEnv([lambda: env])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

#model.learn(total_timesteps=20000)
sys.exit(0)


log_path = os.path.join('Training', 'Logs')
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=4000)



sys.exit(0)

action_space_size = env.action_space
state_space_size = env.observation_space
#SIZE = env.

if start_q_table1 is None:
    # initialize the q-table#
    q_table1 = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                    for iiii in range(-SIZE+1, SIZE):
                        q_table1[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]
    #print("Size of q_table is ", len(q_table1))

else:
    with open(start_q_table1, "rb") as f:
        q_table1 = pickle.load(f)
    #print("Size of q_table is ", len(q_table1))

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.01

rewards_all_episodes = []

# Q-learning algorithm
for episode in range(num_episodes):
    # initialize new episode params
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode): 

        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1) 
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:]) 
        else:
            action = env.action_space.sample()
        
        # Take new action
        new_state, reward, done, info = env.step(action)

        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        # Set new state
        state = new_state

        # Add new reward 
        rewards_current_episode += reward        

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    
    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)


# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000