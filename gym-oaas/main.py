import numpy as np
from maddpg import MADDPG
from buffer import MultiAgentReplayBuffer
#from make_env import make_env

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

#import wandb

#wandb.login()



def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


if __name__ == '__main__':

    #wandb.init()

    #scenario = 'simple'
    scenario = 'simple_adversary'
    #env = make_env(scenario)

    floor_plan_real = [[0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

    agent_skills_2task = [[1,0,0],
                      [1,0,1],
                      [0,1,0],
                      [0,0,1],
                      [1,1,1]]


    env = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)


    n_agents = env.n
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)

    #print(actor_dims)
    #print(critic_dims)
    #sys.exit(0)


    # action space is a list of arrays, assume each agent has same action space
    n_actions = env.action_space[0].n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, 
                           fc1=64, fc2=64,  
                           alpha=0.01, beta=0.01, scenario=scenario,
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, 
                        n_actions, n_agents, batch_size=1024)

    PRINT_INTERVAL = 100
    N_GAMES = 15000 # 50000
    MAX_STEPS = 150
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load_checkpoint()

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False]*n_agents
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            actions = maddpg_agents.choose_action(obs)

            # print("Raw actions are: ",actions)
            # for a in actions:
            #     maximum = np.max(a)
            #     temp_choice_idx = np.where(a == maximum)[0][0]
            #     print("Actual actions are: ", temp_choice_idx)

            obs_, reward, done, info = env.step(actions)
            #print(reward)
            #env.agent_render()

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]*n_agents
                #print(done)
                #sys.exit(0)

            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)

            obs = obs_

            score += reward
            total_steps += 1
            episode_step += 1
            #wandb.log({"Score": score,        
           #"Total Steps": total_steps,        
           #"Episode Step": episode_step})
            

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        #wandb.log({"Score History": score_history[-1],        
           #"Average Score": avg_score})

        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
                #wandb.log({"Best Score": best_score})
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
            #wandb.log({"Iteration": i, "Score":avg_score})