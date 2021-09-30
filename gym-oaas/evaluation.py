import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os

floor_plan1 = [[0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

agent_skills1 = [[0,0],
                [0,1],
                [1,0],
                [1,1]]



env = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan1, agent_skills=agent_skills1)

PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
model = PPO.load(PPO_path, env=env)
evaluate_policy(model, env, n_eval_episodes=10, render=True)

sys.exit(0)
#PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')
#model = PPO.load(PPO_path, env=env)

#evaluate_policy(model, env, n_eval_episodes=10, render=False)
#env.close()
#sys.exit(0)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done: 
        print('info', info)
        break

env.close()