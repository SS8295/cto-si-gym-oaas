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
from lib.dqn import *
from lib.utils import *

import ray
from ray import tune
from ray.tune.registry import register_env
import ray.rllib.agents.ppo as ppo
import shutil
import json
import pathlib
from ray.tune.logger import pretty_print



style.use("ggplot")

floor_plan_real = [[0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0,0],
               [0,1,1,1,1,1,1,1,0,0],
               [0,0,0,0,0,0,0,0,0,0]]

agent_skills_2task = [[1,1,1],
                      [1,1,1],
                      [1,1,1]
                      ]

single_agent = [[0,1,0]]

def env_creator(env_config):
    environment = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)
    return environment  # return an env instance

ray.init()
register_env("stelios", env_creator)
config = ppo.DEFAULT_CONFIG.copy()
config = {"framework": "torch", 
            "env": "stelios"}
trainer = ppo.PPOTrainer(config=config, env="stelios")
# Can optionally call trainer.restore(path) to load a checkpoint.

analysis = ray.tune.run(
    ppo.PPOTrainer,
    config=config,
    local_dir='PPO_SINGLE',
    stop={"training_iteration": 10},
    checkpoint_at_end=True)

checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean"),
    metric="episode_reward_mean")
last_checkpoint = analysis.get_last_checkpoint()
agent = ppo.PPOTrainer(config=config, env='stelios')
agent.restore('PPO_SINGLE')

for i in range(10):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

sys.exit(0)


def train():
    """
    Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
    :param stop_criteria: Dict with stopping criteria.
        See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
    :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
        See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
    """
    analysis = tune.run("PPO", name="stelios", config=config, local_dir='saved_train', stop={"training_iteration": 3}, checkpoint_at_end=True)
    # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
    checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean'),
                                                       metric='episode_reward_mean', mode='max')
    # retriev the checkpoint path; we only have a single checkpoint, so take the first one
    checkpoint_path = checkpoints[0][0]
    return checkpoint_path, analysis

def load(path):
    """
    Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
    :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    """
    agent = ppo.PPOTrainer(config=config, env="stelios")
    agent.restore(path)

def test(self):
    """Test trained agent for a single episode. Return the episode reward"""
    # instantiate env class
    env = self.env_class(self.env_config)

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = self.agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    return episode_reward

ray.init()
register_env("stelios", env_creator)
config = {"framework": "torch", 
            "env": "stelios"}
train()


# ray.shutdown()
# ray.init(ignore_reinit_error=True)
# save_dir = "tmp/ppo/oaas"
# stop_criteria={"training_iteration": 10}
# # shutil.rmtree(save_dir, ignore_errors=True, onerror=None)
# register_env("stelios", env_creator)
# config = {"framework": "torch", "env": "stelios"}
# trainer = tune.run("PPO", name="stelios", config=config)
# checkpoint = trainer.save()
# print("checkpoint saved at", checkpoint)
sys.exit(0)

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   if i % 100 == 0:
       #print(result)
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

sys.exit(0)


# ray.init()
# egister_env("stelios", env_creator)
# config = {"framework": "torch", "env": "stelios"}
# tune.run("PPO", name="stelios", config=config)

# config = ppo.DEFAULT_CONFIG.copy()
# config["log_level"] = "WARN"
# policy = agent.get_policy()
# model = policy.model
# # print(model)

N_ITER = 30
# s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"

results = []
episode_data = []
episode_json = []

# for n in range(N_ITER):
#     result = agent.train()
#     results.append(result)
    
#     episode = {'n': n, 
#                'episode_reward_min': result['episode_reward_min'], 
#                'episode_reward_mean': result['episode_reward_mean'], 
#                'episode_reward_max': result['episode_reward_max'],  
#                'episode_len_mean': result['episode_len_mean']
#               }
    
#     episode_data.append(episode)
#     episode_json.append(json.dumps(episode))
#     file_name = agent.save(checkpoint_root)
#     print(f'{n+1:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}, len mean: {result["episode_len_mean"]:8.4f}. Checkpoint saved to {file_name}')

policy = agent.get_policy()
model = policy.model

pprint.pprint(model.variables())
pprint.pprint(model.value_function())

print(model)


#tune.run("PPO", name="stelios", config=config)


print("Creating Environment...")
env = gym.make('gym_oaas:oaas-v0', floor_plan=floor_plan_real, agent_skills=agent_skills_2task)
print("Environment Created!")

episode_reward = []
done = False
obs = env.reset()
#print(env.agent_skills)
#sys.exit(0)
while not done:
    action = 6 #env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(action_to_schedule(action))
    episode_reward.append(reward)
    env.render()

#env.close()
print("Sum of all rewards = ", sum(episode_reward))
plt.plot(episode_reward)
plt.show()

# Observation should be dictionary instead of matrix
# Reward should be dictionay