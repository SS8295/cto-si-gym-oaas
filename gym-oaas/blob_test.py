# Imports for off-the-shelf environment

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import sys

style.use("ggplot")

SIZE = 10
HM_EPISODES = 25000

MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

epsilon = 0.1
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY

SHOW_EVERY = 3000  # how often to play through env visually.

start_q_table1 = 'qtable-1626214277.pickle' # None or Filename
start_q_table2 = 'qtable-1626214279.pickle'

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table1 is None:
    # initialize the q-table#
    q_table1 = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                    for iiii in range(-SIZE+1, SIZE):
                        q_table1[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]
    #print("Size of q_table is ", len(q_table1))

else:
    with open(start_q_table1, "rb") as f:
        q_table1 = pickle.load(f)
    #print("Size of q_table is ", len(q_table1))


if start_q_table2 is None:
    # initialize the q-table#
    q_table2 = {}
    for i in range(-SIZE+1, SIZE):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                    for iiii in range(-SIZE+1, SIZE):
                        q_table2[((i, ii), (iii, iiii))] = [np.random.uniform(-5, 0) for i in range(4)]

else:
    with open(start_q_table2, "rb") as f:
        q_table2 = pickle.load(f)

# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []

for episode in range(HM_EPISODES):
    player1 = Blob()
    player2 = Blob()
    food = Blob()
    enemy = Blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs1 = (player1-food, player1-enemy)
        obs2 = (player2-food, player2-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action1 = np.argmax(q_table1[obs1])
            action2 = np.argmax(q_table2[obs2])
        else:
            action1 = np.random.randint(0, 4)
            action2 = np.random.randint(0, 4)
        # Take the action!
        player1.action(action1)
        player2.action(action2)

        #### MAYBE ###
        #enemy.move()
        #food.move()
        ##############

        if player1.x == enemy.x and player1.y == enemy.y:
            reward1 = -ENEMY_PENALTY
        elif player1.x == food.x and player1.y == food.y:
            reward1 = FOOD_REWARD
        else:
            reward1 = -MOVE_PENALTY

        if player2.x == enemy.x and player2.y == enemy.y:
            reward2 = -ENEMY_PENALTY
        elif player2.x == food.x and player2.y == food.y:
            reward2 = FOOD_REWARD
        else:
            reward2 = -MOVE_PENALTY

        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs1 = (player1-food, player1-enemy)
        new_obs2 = (player2-food, player2-enemy)

        max_future_q1 = np.max(q_table1[new_obs1])
        current_q1 = q_table1[obs1][action1]

        max_future_q2 = np.max(q_table2[new_obs2])
        current_q2 = q_table2[obs2][action2]


        if reward1 == FOOD_REWARD:
            new_q1 = FOOD_REWARD
        else:
            new_q1 = (1 - LEARNING_RATE) * current_q1 + LEARNING_RATE * (reward1 + DISCOUNT * max_future_q1)
        q_table1[obs1][action1] = new_q1

        if reward2 == FOOD_REWARD:
            new_q2 = FOOD_REWARD
        else:
            new_q2 = (1 - LEARNING_RATE) * current_q2 + LEARNING_RATE * (reward2 + DISCOUNT * max_future_q2)
        q_table2[obs2][action2] = new_q2

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player1.x][player1.y] = d[PLAYER_N]  # sets the player tile to blue
            env[player2.x][player2.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((500, 500))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward1 == FOOD_REWARD or reward1 == -ENEMY_PENALTY or reward2 == FOOD_REWARD or reward2 == -ENEMY_PENALTY :  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        episode_reward += reward1 + reward2
        if reward1 == FOOD_REWARD or reward1 == -ENEMY_PENALTY or reward2 == FOOD_REWARD or reward2 == -ENEMY_PENALTY:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table1, f)

with open(f"qtable-{int(time.time()+1)}.pickle", "wb") as f:
    pickle.dump(q_table2, f)