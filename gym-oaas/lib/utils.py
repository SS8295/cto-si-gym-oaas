from os import path
import os
import numpy as np
import cv2
import sys
import random
import os
import copy


task_dict_map = {
    0 : 'bathroom',
    1 : 'cachier',
    2 : 'pickup'
}


# DeepRM MAPPING #

def skillsToDistanceCalc(skills, task_dict, map):

    for col in range(len(skills)):
        for row in range(len(skills[0])):
            if skills[col][row] == 0:
                skills[col][row] = None

            else:
                check_skill_flag = False
                for key in task_dict:
                    if map[row] in key:
                        check_skill_flag = True
                if check_skill_flag == False:
                    skills[col][row] = None
    return skills

# OS # 

def doesFileExist(filePathAndName):
    return os.path.exists(filePathAndName)

# RL #

def env_reward(task_dict):

    for element in task_dict:
        pass
    pass

# TASKS # 

task_arrival_probabilities = {  'bathroom' : 0.5,
                                'cashier' : 0.5,
                                'pickup' : 0.5
                                }

def duration_task(dist, w):

    '''Accepts a list of distances from a target + the integer work of the target'''

    if dist == []:
        return 0

    d_sort = sorted(dist)
    timer = 0
    
    i = 0
    while w > 0:
        w -= i
        if w==0:
            return timer
        timer +=1
        while i <= len(d_sort)-1 and timer >= d_sort[i]:
            i += 1
        
    return timer

def bathroom_task_check():
    bathroom_task_prob = task_arrival_probabilities['bathroom']
    return random.random() < bathroom_task_prob

def cachier_task_check():
    cachier_task_prob = task_arrival_probabilities['cashier']
    return random.random() < cachier_task_prob

def pickup_task_check():
    pickup_task_prob = task_arrival_probabilities['pickup']
    return random.random() < pickup_task_prob

# RENDERING #

fps = 500

def draw_grid(grid, counter):

    img = grid #Image.fromarray(state, 'RGB')
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 300,300) 
    cv2.imshow("image", img)
    width = 300 # int(img.shape[1] * percent/ 100)
    height = 300 # int(img.shape[0] * percent/ 100)
    dim = (width, height)
    
    temp_img = cv2.resize(img, dim, interpolation =cv2.INTER_AREA)
    cv2.imwrite('./video/frame{}.jpg'.format(counter), temp_img)
    cv2.waitKey(fps)

def rescale_frame(frame, percent=3000):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def showPartial(grid): # def render(self, mode='human'):
        img = grid #Image.fromarray(self.state, 'RGB')
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 300,300) 
        cv2.imshow("image", img)
        frame150 = rescale_frame(img, percent=3000)
        #cv2.imwrite('video/output'+str(self.time_counter)+'.jpg',frame150)
        cv2.waitKey(fps)

# GRIDWORLD # 

def select_subgrid(coords, scope, grid_state):

    #global grid_state

    floor_plan_final = np.zeros((len(grid_state)+2*scope,len(grid_state[0])+2*scope,3))
    floor_plan_final[scope:len(grid_state)+scope,scope:len(grid_state[0])+scope,:] = grid_state

    x_max = coords[0]+2*scope+1
    x_min = coords[0]
    y_max = coords[1]+2*scope+1
    y_min = coords[1]

    #showPartial(floor_plan_final[x_min : x_max , y_min : y_max])

    return floor_plan_final[x_min : x_max , y_min : y_max] #.flatten()

def create_wall_dictionary(input_grid):
    
    empty_space = set()
    non_empty_space = set()

    for i in range(len(input_grid)):
        for j in range(len(input_grid[0])):
            if input_grid[i][j] == 1:
                non_empty_space.add((i,j))
            elif input_grid[i][j] == 0:
                empty_space.add((i,j))

    return empty_space,non_empty_space

def add_walls(grid, obstacle_set):

    for coods in obstacle_set:
        grid[coods[0]][coods[1]][:] = 255
    
    return grid

def refresh_grid(full_grid, employees, tasks):

    complete_grid = copy.deepcopy(full_grid)

    # # Render Agents
    if not employees:
        pass
    else:
        for i in employees:
            complete_grid[employees[i].x][employees[i].y] = employees[i].color

    # # Render Tasks
    if not tasks:
        pass
    else:
        for i in tasks:
            complete_grid[tasks[i].x][tasks[i].y] = tasks[i].color

    return complete_grid

# MISCELLANEOUS #

def action_to_schedule(n, num_agents):
    if n == 0:
        #print([[0,0,0],[0,0,0],[0,0,0]])
        return [[1,0,0]] * num_agents
    else:

        nums = []
        while n:
            n, r = divmod(n, 4)
            nums.append(str(r))

        temp_str = ''.join(reversed(nums)).zfill(num_agents)
        # print(temp_str)
        # sys.exit(0)
        schedule = []
        for idx in temp_str:
            temp_schedule = [0] * 4
            temp_schedule[int(idx)] = 1
            if int(idx) == 3:
                temp_schedule = [0,0,0]
            schedule.append(temp_schedule[0:3])
        # print("This is the schedule: ",schedule[-3:])
        # sys.exit(0)
        # print("Current Schedule = ",schedule[-3:])
        return schedule[-num_agents:]

def print_matching(action_list, num_agents):

    for i in range(len(action_list)):

        if action_list[i] == [0,0,0]:
            print("Agent ",i,"                                     stay where you are")
        else:
            for index in range(len(action_list[i])):
                if action_list[i][index] == 1:
                    print("Agent ",i," matched to                                  ", task_dict_map[index])
                    return index

def find_index(array):

    for k,v in enumerate(array):
        if v == 1:
            return k
        else:
            return 

# def freeze_step(action_int, task_dict, employee_dict, horizon, counter, render_flag):

#     reward = []
#     done = []
#     freezed_reward = 0

#     for i in range(horizon):
#         counter += 1
#         if render_flag == True:
#             print("STEP ",i,"/3")






    

#     return counter

