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
# from max_flow import FindMaximumMatching
# from max_flow import make_circuit_video

# import wandb

random.seed(0)

floor_plan_real = [[0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

agent_skills_2task = [[1,0,1],
                      [0,0,1],
                      [1,1,0]
                      ]

single_agent = [[0,1,0]]

# wandb.init(project="MaxFlow Reward Tracking")
print("Creating Environment...")
env = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)
print("Environment Created!")

print(env.task_dict_map)
print(env.task_dict)
print("Possible Schedules = ",env.possible_schedules)
print("Possible Actions = ",env.action_sets)

sys.exit(0)

#random_scheduler = np.zeros((env.n, len(env.task_dict)))
#random_scheduler = [[0] * len(env.task_dict)] * env.n
random_scheduler = [[0] * len(env.task_dict) for i in range(env.n)]
print(random_scheduler)
#sys.exit(0)

for action_set in range(len(env.action_sets)):

    if not env.action_sets[action_set]:
        continue
    scheudled_task = random.choice(env.action_sets[action_set])
    print("Schedule picked: ", scheudled_task)
    print("action_set: ", action_set)
    if env.task_dict[env.task_dict_map[scheudled_task]].max_agents == 0:
        for rest in range(action_set, len(env.action_sets)):
            env.possible_schedules[rest][scheudled_task] = None
            if scheudled_task in env.action_sets[rest]:
                env.action_sets[rest].remove(scheudled_task)

        continue

    random_scheduler[action_set][scheudled_task] = 1
    print(random_scheduler)
    print("--------------")
    env.task_dict[env.task_dict_map[scheudled_task]].max_agents -= 1
    
    if env.task_dict[env.task_dict_map[scheudled_task]].max_agents == 0:
        for rest in range(action_set, len(env.action_sets)):
            env.possible_schedules[rest][scheudled_task] = None
            if scheudled_task in env.action_sets[rest]:
                env.action_sets[rest].remove(scheudled_task)

print('Random Scheduler = ',random_scheduler)


# RANDOM SCHEDULER #

sys.exit(0)

print("Num Employees = ",len(env.employee_dict))
print("Num Tasks = ", len(env.task_dict))
temp_reward = []
#sys.exit(0)

# -------- Max Flow Scheduler - Baseline ------- #
for iter in range(5):

    env.fetch_new_tasks()

    # print("Reached this and printing data")
    # print(env.employee_dict)
    # print(env.task_dict)

    num_agents = len(env.agent_skills)
    num_possible_tasks = len(env.agent_skills[0])

    max_flow_schedule = np.zeros((num_agents,num_possible_tasks))
    vertices = []
    edges = []

    for key in env.employee_dict:
        vertices.append(key)
    for key in env.task_dict:
        vertices.append(key)

    for agent in env.employee_dict:
        for skill in range(len(env.employee_dict[agent].skill)):
            #print("Agent = ", agent)
            #print("Skill = ", skill)
            #print(env.employee_dict[agent].skill[skill])
            if env.employee_dict[agent].skill[skill] == 1 and env.task_dict_map[skill] in env.task_dict:
                edges.append({agent,env.employee_dict[agent].task_dict_map[skill]})

    print("Vertices: ",vertices)
    print("Edges: ",edges)

    f = FindMaximumMatching(edges, vertices)
    f.find_maximum_matching()
    # print("Schedule complete: ",f.matching)

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
        observation, reward, done, info = env.schedule_step(max_flow_schedule)
        temp_reward.append(reward)
        #env.render()
        #env.create_tasks

#env.close()
plt.plot(temp_reward)
plt.show()
#make_circuit_video('animation.gif', fps=1)
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