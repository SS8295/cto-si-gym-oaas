import numpy as np
import sys

def select_subgrid(floor_plan, coords, scope):

    grid_height = len(floor_plan)
    grid_length = len(floor_plan[0])

    floor_plan_final = np.zeros((grid_height+2*scope,grid_length+2*scope))
    floor_plan_final[scope:grid_height+scope,scope:grid_length+scope] = floor_plan
    
    x_max = coords[0]+2*scope+1
    x_min = coords[0]
    y_max = coords[1]+2*scope+1
    y_min = coords[1]

    return floor_plan_final[x_min : x_max , y_min : y_max]

floor_plan_real = [[10,20,30,40,50,60,70,80,90,100],
               [11,21,31,41,51,61,71,81,91,101]]


# floor_plan_real = [[0,0,0,0,0,0,0,0,0,0],
#                [0,1,1,1,1,1,1,1,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,1,1,1,1,1,1,1,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,0,0,0,0,0,0,0,0,0],
#                [0,1,1,1,1,1,1,1,0,0],
#                [0,0,0,0,0,0,0,0,0,0]]

employee_tuple = (1,7)
employee_tuple2 = (9,9)
employee_scope = 1

print(select_subgrid(floor_plan_real, employee_tuple, employee_scope))