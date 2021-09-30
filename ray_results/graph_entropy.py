import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter


smoothingWeight = 1

df = pd.read_csv('./PPO_oaas-v0_2021-09-23_22-22-160w3hfgeh/progress.csv')

rew_temp = df['episode_reward_mean']
entropy = df['info/learner/default_policy/learner_stats/entropy']
x_labels =  df['timesteps_total']/1000

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

fig, ax = plt.subplots()

ax.plot(x_labels, rew_temp,'-.', label='Exact',alpha=0.4)
ax.plot(x_labels, smooth(rew_temp, .8), label='Smooth',c='k')

#plt.plot(x_labels, smooth(rew_temp, .8), c='red')
#plt.legend("f",'sfs')

plt.xlabel("# (thousand) Iterations")
plt.ylabel("Mean Reward")
#plt.legend("H",'J')
leg = ax.legend()

ax2=ax.twinx()
ax2.plot(x_labels, entropy,'-.', label='Entropy',alpha=0.4,c='tab:orange')
ax2.set_ylabel("Entropy")
leg = ax2.legend()


fig.savefig('demo.png')