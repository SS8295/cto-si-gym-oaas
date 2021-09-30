import random
import copy

class BathroomTask:

    def __init__(self, empty_space):
        
        (self.x, self.y) = random.choice(tuple(empty_space)) # (8,9) #
        self.coords = (self.x, self.y)
        self.color = (255,0,0)
        self.max_agents = 5

        self.work_rem = 10
        self.te = copy.deepcopy(self.work_rem)
        self.sum_tw = 0
        self.sum_te = 0

        self.penalty = 10
        self.priority = 0