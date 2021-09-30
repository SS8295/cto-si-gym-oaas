import random
import copy

class CachierTask:

    def __init__(self, empty_space):
        
        (self.x, self.y) = random.choice(tuple(empty_space))
        self.coords = (self.x, self.y)
        self.color = (0,255,0)
        self.max_agents = 3
        
        self.work_rem = 10
        self.te = copy.deepcopy(self.work_rem)
        self.sum_tw = 0
        self.sum_te = 0

        self.penalty = 1
        self.priority = 6