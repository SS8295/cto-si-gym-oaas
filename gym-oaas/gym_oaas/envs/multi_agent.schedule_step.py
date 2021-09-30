    def schedule_step(self, task_action):


        # Initialize variable to track a_star dictionary lookup frequency
        # self.a_star_dict_ref
    
        # Time in terms of timesteps
        self.time_counter +=1

        # List of Rewards and Done Flags
        observation = []
        reward = []
        done = [] 
        num_agents = len(task_action)

        for i in range(num_agents):
            
            self.update_pathfinding_dict()
            employee_i = self.employee_dict["employee"+str(i)]
            
            print(employee_i.x)
            print(employee_i.y)

            employee_i.action_astar(task_action[i], self.agent_task_distances[i], self.task_dict, task_dict_map, self.pathfinding_dict, self.obstacles_set, self.numpy_grid)
            employee_i.x # temp x coords of agent i
            employee_i.y # temp y coords of agent i
            self.refresh_state()
            observation.append(employee_i.flat_obs)
            self.update_pathfinding_dict()

            for task in self.task_dict:
                #print("Work for task: ",task," is: ",self.task_dict[task].work)
                if self.task_dict[task].work <= 0:
                    del self.task_dict[task]

            for task in self.task_dict:
                if (employee_i.x, employee_i.y) == (self.task_dict[task].x, self.task_dict[task].y) and self.agent_skills[i][task_dict_inverse_map[task]] == 1:
                    
                    self.task_dict[task].work -= 1
                    if self.task_dict[task].work <= 0:
                        del self.task_dict[task]
                        break
  
        for task in self.task_dict:
            reward.append(self.task_dict[task].penalty)
        
        # wandb.log({'total_penalty': sum(reward)})
        print("Total penalty = ", reward)   
        print("reward = ", sum(reward))

        for empl in range(len(self.employee_dict)):
            done.append(self.employee_dict["employee"+str(empl)].done)

        info = {}

        return observation, sum(reward), done, info