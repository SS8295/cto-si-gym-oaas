import sys
import cv2
import copy
import time
import json  
import random
import pickle
import glob, os, os.path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.stats import poisson
from typing import DefaultDict
from datetime import datetime
import math

import gym
from gym import Env
from gym.spaces import Discrete, Box
from gym import error, spaces, utils
from gym.utils import seeding

from lib.employee import Employee
from lib.astar import astar
from lib.bathroom_task import BathroomTask
from lib.cachier_task import CachierTask
from lib.pickup_task import PickupTask
from lib.utils import *
#from lib.scheduler import duration_task
# import wandb

style.use("ggplot")
# --------------------------------------------

pathfinding_dict = {}
temp_freq = [0]

task_dict_map = {
    0 : 'bathroom',
    1 : 'cachier',
    2 : 'pickup'
}

task_dict_inverse_map = {
    'bathroom': 0,
    'cachier': 1,
    'pickup': 2
}

# ------------ OPENAI_GYM_ENVIRONMENT BEGIN ------------ #

class OaasEnv(Env):
    
    def __init__(self, floor_plan, agent_skills):

        # Initialization
        self.time_counter = 0
        self.task_dict_inverse_map = { 'bathroom': 0,
                                       'cachier': 1,
                                       'pickup': 2   }
        self.task_dict_map = {  0 : 'bathroom',
                                1 : 'cachier',
                                2 : 'pickup'   }
        # Grid variables
        self.grid_height = len(floor_plan)
        self.grid_width = len(floor_plan[0])
        self.numpy_grid = copy.deepcopy(floor_plan)
        self.empty_space_set , self.obstacles_set = create_wall_dictionary(floor_plan)
        self.complete_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        self.complete_grid = add_walls(self.complete_grid, self.obstacles_set)
        self.full_grid = copy.deepcopy(self.complete_grid)

        # RL variables
        self.agent_skills = copy.deepcopy(agent_skills)
        self.n = len(self.agent_skills)
        self.action_space = Discrete((len(self.task_dict_map)+1) ** self.n)
        self.observation_space = Box(low=-1.0, high=100.0, shape=(12, ), dtype=np.float32)

        # Load pathfinding dictionary
        self.pathfinding_dict = {}
        if doesFileExist('pathfinding_cache.pkl'):
            a_file = open("pathfinding_cache.pkl", "rb")
            self.pathfinding_dict = pickle.load(a_file)
            a_file.close()

        # Summon Employees
        self.employee_dict = {}
        for i in range(self.n):
            self.employee_dict["employee"+str(i)] = Employee(self.agent_skills[i], self.empty_space_set, self.complete_grid)
            employee_i = self.employee_dict["employee"+str(i)]
            self.complete_grid[employee_i.x][employee_i.y] = employee_i.color
            # self.action_space.append(self.employee_dict["employee"+str(i)].action_space)
            # self.observation_space.append(self.employee_dict["employee"+str(i)].observation_space)

        #Summon Tasks
        self.task_dict = {}
        self.fetch_new_tasks()

        # Create worklist of all tasks
        if len(self.task_dict)>0:
            self.worklist = [0 for _ in self.task_dict_map]
            for task in self.task_dict:
                self.worklist[self.task_dict_inverse_map[task]] = self.task_dict[task].work
        self.complete_grid = refresh_grid(self.full_grid, self.employee_dict, self.task_dict)
     
        self.agent_task_distances = copy.deepcopy(self.agent_skills)

        for row in range(len(self.agent_task_distances)):
            for col in range(len(self.agent_task_distances[0])):
                if self.task_dict_map[col] not in self.task_dict:
                    self.agent_task_distances[row][col] = -1
                else:
                    self.agent_task_distances[row][col] = 0
        self.update_pathfinding_dict()

        print("Initial Agent-Task Distances = ",self.agent_task_distances)
    
    def step(self, task_action):
        
        #print(self.freeze_step(task_action))
        return self.freeze_step(task_action)

    def freeze_step(self, action):

        reward = []
        done = [] 

        freezed_reward = 0
        self.frozen_task_dict = copy.deepcopy(self.task_dict)
        self.locked_task_dict = copy.deepcopy(self.task_dict)
        self.frozen_worklist = copy.deepcopy(self.worklist)

        for i in range(30):
            # self.render()
            # Time in terms of timesteps
            self.time_counter +=1
            # print("printing task action",action)
            task_action = action_to_schedule(action)
            # print("This is task aciton: ",task_action)
            num_agents = len(task_action)
        
            # List of Rewards and Done Flags
            # observation = []

            # self.update_pathfinding_dict()
            #print(self.agent_task_distances)

            for i in range(num_agents):
                
                self.update_pathfinding_dict()
                # print("Agent ", i, " with skills: ", self.employee_dict["employee"+str(i)].skill, " assigned to task: ", task_action[i])
                employee_i = self.employee_dict["employee"+str(i)]
                employee_i.action_astar(task_action[i], self.agent_task_distances[i], self.locked_task_dict, task_dict_map, self.pathfinding_dict, self.obstacles_set, self.numpy_grid)
                employee_i.x # temp x coords of agent i
                employee_i.y # temp y coords of agent i
                self.complete_grid = refresh_grid(self.full_grid, self.employee_dict, self.frozen_task_dict)

                for task in self.frozen_task_dict:
                    # print("Work for task: ",task," is: ",self.task_dict[task].work)
                    if self.frozen_task_dict[task] in self.frozen_task_dict:
                        self.frozen_worklist[self.task_dict_inverse_map[task]] = self.frozen_task_dict[task].work
                        if self.frozen_task_dict[task].work <= 0:
                            self.frozen_worklist[self.task_dict_inverse_map[task]] = 0
                            del self.frozen_task_dict[task]

                for task in self.frozen_task_dict:
                    if (employee_i.x, employee_i.y) == (self.frozen_task_dict[task].x, self.frozen_task_dict[task].y) and self.agent_skills[i][task_dict_inverse_map[task]] == 1:
                        self.frozen_task_dict[task].work -= 1
                        if self.frozen_task_dict[task] in self.frozen_task_dict:
                            self.frozen_worklist[self.task_dict_inverse_map[task]] = self.frozen_task_dict[task].work
                        if self.frozen_task_dict[task].work <= 0:
                            for agent in range(len(self.agent_task_distances)):
                                self.agent_task_distances[agent][self.task_dict_inverse_map[task]] = -1
                            self.frozen_worklist[self.task_dict_inverse_map[task]] = 0
                            del self.frozen_task_dict[task]
                            break

            # print("Internal done flag is: ", done)
            for task in self.frozen_task_dict:
                reward.append(-math.e ** -(self.frozen_task_dict[task].priority+1/self.frozen_task_dict[task].penalty-2))
            
            for agent in self.employee_dict:
                reward.append(-self.employee_dict[agent].wrong_action_penalty)
            #print(agent, " wrong action: ",self.employee_dict[agent].wrong_action_penalty)  
            #print("Total task and scheduling penalty = ", reward) 
            #print("reward = ", sum(reward))
            # for empl in range(len(self.employee_dict)):
            #     done.append(self.employee_dict["employee"+str(empl)].done)
            info = {}

            observation = copy.deepcopy(self.agent_task_distances)
            observation.append(self.frozen_worklist)
            # print(sum(reward))
            observation = np.array(observation)

            #print("Agent Task Distances:", self.agent_task_distances)
            #print("Workloads:", self.frozen_worklist)


            #self.fetch_new_tasks()
            for row in range(len(self.agent_task_distances)):
                for col in range(len(self.agent_task_distances[0])):
                    if self.task_dict_map[col] not in self.frozen_task_dict:
                        self.agent_task_distances[row][col] = -1
                    else:
                        self.agent_task_distances[row][col] = 0

            if self.time_counter >= 200: # len(self.task_dict) == 0 or 
                done = True
            else:
                done = False

        freezed_reward = sum(reward)
        #print(freezed_reward)
        self.update_pathfinding_dict
        self.task_dict = copy.deepcopy(self.frozen_task_dict)
        self.fetch_new_tasks()

        return observation.flatten() , freezed_reward, done, info

    def reset(self):

        self.time_counter = 0
        # Grid variables
        self.complete_grid = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
        self.complete_grid = add_walls(self.complete_grid, self.obstacles_set)
        self.full_grid = copy.deepcopy(self.complete_grid)

        # Summon employees and tasks
        self.employee_dict = {}
        self.task_dict = {}
    
        for i in range(self.n):
            self.employee_dict["employee"+str(i)] = Employee(self.agent_skills[i], self.empty_space_set, self.complete_grid)
            employee_i = self.employee_dict["employee"+str(i)]
            self.complete_grid[employee_i.x][employee_i.y] = employee_i.color
            # self.action_space.append(self.employee_dict["employee"+str(i)].action_space)
            # self.observation_space.append(self.employee_dict["employee"+str(i)].observation_space)


        self.fetch_new_tasks()
        self.frozen_task_dict = copy.deepcopy(self.task_dict)
        self.worklist = [-1 for _ in range(len(self.task_dict_map))]

        for task in self.task_dict:
            self.worklist[self.task_dict_inverse_map[task]] = self.task_dict[task].work
        self.complete_grid = refresh_grid(self.full_grid, self.employee_dict, self.frozen_task_dict)
     
        self.agent_task_distances = copy.deepcopy(self.agent_skills)
        for row in range(len(self.agent_task_distances)):
            for col in range(len(self.agent_task_distances[0])):
                if self.task_dict_map[col] not in self.task_dict:
                    self.agent_task_distances[row][col] = -1
                else:
                    self.agent_task_distances[row][col] = 0
        # print(self.agent_task_distances)

        self.update_pathfinding_dict()

        # print("Initial Agent-Task Distances = ",self.agent_task_distances)

        observation = copy.deepcopy(self.agent_task_distances)
        observation.append(self.worklist)
        observation = np.array(observation)
        return observation.flatten()

    def render(self, mode='human'): # def render(self, mode='human'):

        print("Time = ", self.time_counter)
        print("Tasks remaining = ", self.task_dict)
        # print("Work left for each task is:")
        # for task_ in self.task_dict:
        #     print(task_,": ",self.task_dict[task_].work)
        self.complete_grid = refresh_grid(self.full_grid, self.employee_dict, self.frozen_task_dict)
        draw_grid(self.complete_grid, self.time_counter)
    
    def update_pathfinding_dict(self):

        for i in range(len(self.agent_skills)):

            for j in range(len(self.agent_skills[0])):
                if self.agent_task_distances[i][j] != -1:
                    start = (self.employee_dict["employee"+str(i)].x,self.employee_dict["employee"+str(i)].y)
                    end = (self.task_dict[task_dict_map[j]].x,self.task_dict[task_dict_map[j]].y)
                    start_end_coords = (start[0],start[1],end[0],end[1])
                    if start_end_coords in self.pathfinding_dict:
                        path = self.pathfinding_dict[start_end_coords][1]
                        path_len = self.pathfinding_dict[start_end_coords][0]
                        self.agent_task_distances[i][j] = len(path)-1

                        #print(self.agent_task_distances)
                        # self.a_star_dict_ref += 1
                        # temp_freq.append(self.a_star_dict_ref/len(temp_freq))

                    else:
                        path = astar(self.numpy_grid, start, end)
                        path_len = len(path)
                        self.agent_task_distances[i][j] = len(path)-1
                        #print(self.agent_task_distances)
                        self.pathfinding_dict[start_end_coords] = (path_len,path)
                        # temp_freq.append(self.a_star_dict_ref/len(temp_freq))
        
    def fetch_new_tasks(self):
        
        if 'bathroom' not in self.task_dict.keys():
            if bathroom_task_check():
                self.task_dict['bathroom'] = BathroomTask(self.empty_space_set)
                task_i = self.task_dict['bathroom']
                self.complete_grid[task_i.x][task_i.y] = task_i.color
        if 'cachier' not in self.task_dict.keys():
            if cachier_task_check():
                self.task_dict['cachier'] = CachierTask(self.empty_space_set)
                task_i = self.task_dict['cachier']
                self.complete_grid[task_i.x][task_i.y] = task_i.color
        if 'pickup' not in self.task_dict.keys():
            if pickup_task_check():
                self.task_dict['pickup'] = PickupTask(self.empty_space_set)
                task_i = self.task_dict['pickup']
                self.complete_grid[task_i.x][task_i.y] = task_i.color
        
    def close(self):

        size = (300,300)
        img_array = []
        sorted_filenames = []
        for filename in glob.glob('video/*.jpg'):
            sorted_filenames.append(filename)
        sorted_filenames.sort(key=os.path.getctime)

        #sorted_filenames.sort(key = lambda x: int(x[:-4]))
        #print(sorted_filenames)

        for filename in range(len(sorted_filenames)):

            img = cv2.imread(sorted_filenames[filename])
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)

        #img_array.sort(key = lambda x: int(x[:-4]))

        now = str(time.time())
        out = cv2.VideoWriter('video/output'+now+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 1, size)
        
        filelist = glob.glob(os.path.join('video/', "*.jpg"))
        #print(filelist)
        # for f in filelist:
        #     os.remove(f)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        # plt.plot(temp_freq)
        # plt.show()

    
        a_file = open("pathfinding_cache.pkl", "wb")
        pickle.dump(self.pathfinding_dict, a_file)
        a_file.close()