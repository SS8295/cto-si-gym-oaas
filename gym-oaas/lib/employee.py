import random
from gym.spaces import Discrete, Box
import numpy as np
from lib.utils import select_subgrid
from lib.astar import astar
import sys
from lib.utils import draw_grid

class Employee:

    def __init__(self,skillset, empty_space, complete_grid):

        self.task_dict_map = {
            0 : 'bathroom',
            1 : 'cachier',
            2 : 'pickup'
        }

        self.skill = skillset
        self.color=(220,220,220)
        color_list = list(self.color)
        for i in range(len(self.skill)):
            if self.skill[i] == 1:
                color_list[i] = color_list[i] * self.skill[i]
            elif self.skill[i] == 0:
                color_list[i] = 50
        self.color = tuple(color_list)

        (self.x, self.y) = random.choice(tuple(empty_space))
        self.coords = (self.x, self.y)
        self.action_space = Discrete(5)
        self.observation_scope = 5
        self.observation_size = (2*self.observation_scope + 1) * (2*self.observation_scope + 1) * 3
        self.observation_space = Box(0,255, shape=(self.observation_size,))
        self.observation = select_subgrid(self.coords, self.observation_scope, complete_grid)
        self.flat_obs = np.array(self.observation).flatten()
        self.reward = 0
        self.wrong_action_penalty = 0
        self.done = False

    def action(self, choice, floor_plan_obstacles, floor_plan_global):

        # MADDPG
        #maximum = np.max(choice)
        #temp_choice_idx = np.where(choice == maximum)[0][0]

        # Non-MADDPG    
        temp_choice_idx = choice

        if temp_choice_idx == 0: # NoP
            self.move(0, 0, floor_plan_obstacles, floor_plan_global)
        elif temp_choice_idx == 1: # down
            self.move(1, 0, floor_plan_obstacles, floor_plan_global)
        elif temp_choice_idx == 2: # right
            self.move(0, 1, floor_plan_obstacles, floor_plan_global)
        elif temp_choice_idx == 3: # left
            self.move(0, -1, floor_plan_obstacles, floor_plan_global)
        elif temp_choice_idx == 4: # up
            self.move(-1, 0, floor_plan_obstacles, floor_plan_global)

    def action_astar(self, task, distances, task_dict, task_map, path_dictionary ,floor_plan_obstacles, floor_plan_global):

        tmp = max(task)                         # Take argmax of distribution
        index = task.index(tmp)

        if task == [0,0,0] or distances[index] != -1:
            self.wrong_action_penalty = 0
        else:
            self.wrong_action_penalty = 1


        if all(v == 0 for v in task) or distances[index] == -1 or (self.x, self.y) == (task_dict[task_map[index]].x, task_dict[task_map[index]].y):
            return
        #print("Agent currently at: ", self.coords)
        next_coords = path_dictionary[(self.x, self.y, task_dict[task_map[index]].x, task_dict[task_map[index]].y)][1][0]
        #print("Next coords are:", next_coords)
        next_x = next_coords[0] - self.x
        next_y = next_coords[1] - self.y
        #print("Relative motion = ", next_x, next_y)
        self.move(next_x, next_y, floor_plan_obstacles, floor_plan_global)
        #print("Agent currently at: ", self.coords)

    def move(self, x, y, floor_plan_obstacles, floor_plan_global):
        
        temp_x = self.x + x
        temp_y = self.y + y

        if (temp_x, temp_y) in floor_plan_obstacles:
            return
        else:
            if temp_x < 0:
                self.x = 0
                self.y = temp_y
                return
            elif temp_x > len(floor_plan_global)-1:
                self.x = len(floor_plan_global)-1
                self.y = temp_y
                return
            if temp_y < 0:
                self.y = 0
                self.x = temp_x
                return
            elif temp_y > len(floor_plan_global[0])-1:
                self.y = len(floor_plan_global[0])-1
                self.x = temp_x
                return
            self.x = temp_x
            self.y = temp_y
            self.coords = (self.x, self.y)
