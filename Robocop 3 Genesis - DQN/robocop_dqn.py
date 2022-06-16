import time
import retro
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import argparse

from algos.agents.dqn_agent import DQNAgent
from algos.models.dqn_cnn import DQNCnn
from algos.preprocessing.stack_frame import preprocess_frame, stack_frame
from datetime import timedelta

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--train', action='store_true', help='Train')
parser.add_argument('--model_epoch',default='1000', type=str, help='Which model to load? Specify the epoch.')
opt = parser.parse_args()

model_epoch = opt.model_epoch

if opt.train:
    print('This is training phase\n')
else:
    print('This is test phase, model: ' + model_epoch + 'th epoch\n')

env = retro.make(game='RoboCop3-Genesis')
env.seed(0)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

print("The size of frame is: ", env.observation_space.shape)
print("No. of Actions: ", env.action_space.n)

possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

def stack_frames(frames, state, is_new=False):
    frame = preprocess_frame(state, (1, -1, -1, 1), 84)
    frames = stack_frame(frames, frame, is_new)

    return frames

INPUT_SHAPE = (4, 84, 84)
ACTION_SIZE = len(possible_actions)
SEED = 0
GAMMA = 0.99           # discount factor
BUFFER_SIZE = 100000   # replay buffer size
BATCH_SIZE = 32        # Update batch size
LR = 0.0001            # learning rate
TAU = 1e-3             # for soft update of target parameters
UPDATE_EVERY = 100     # how often to update the network
UPDATE_TARGET = 10000  # After which thershold replay to be started
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.3          # Ending value of epsilon
EPS_DECAY = 30         # Rate by which epsilon to be decayed

agent = DQNAgent(INPUT_SHAPE, ACTION_SIZE, SEED, device, BUFFER_SIZE, BATCH_SIZE, GAMMA, LR, TAU, UPDATE_EVERY, UPDATE_TARGET, DQNCnn, opt.train, model_epoch)

start_epoch = 0
scores = []
scores_window = deque(maxlen=20)
scores_info = []
scores_window_info = deque(maxlen=20)

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY)

# plt.plot([epsilon_by_epsiode(i) for i in range(1000)])

model_dir = 'model/model_dqn'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

def train(n_episodes=1000):
    """
    Params
    ======
        n_episodes (int): maximum number of training episodes
    """
    env.viewer = None
    start_time_total = time.time()
    for i_episode in range(start_epoch + 1, n_episodes + 1):
        start_time = time.time()
        state = stack_frames(None, env.reset(), True)
        score = 0
        score_info = 0

        eps = epsilon_by_epsiode(i_episode)

        # Punish the agent for not moving forward
        prev_state = {}
        timestamp = 0
        while timestamp < 10000:
            env.render(close=False)
            action = agent.act(state, eps)
            next_state, reward, done, info = env.step(possible_actions[action])
            score_info = info['score']

            timestamp += 1

            if timestamp > 1:
                if (prev_state['ammo'] < info['ammo']):
                    reward += 5
                # Punish the agent for wasting ammo
                if (info['ammo'] == 0):
                    reward -= 5
                if (info['ammo'] == 0 and info['health'] == 0):
                    reward -= 20
                if (prev_state['ammo'] > info['ammo'] and prev_state['score'] < info['score']):
                    reward += 700
                if (info['score'] == 0):
                    reward -= 5

            prev_state = info
            score += reward

            next_state = stack_frames(state, next_state, False)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        scores_info.append(score_info)
        scores_window_info.append(score_info)

        time_taken = str(timedelta(seconds=(time.time() - start_time)))
        print('\rEpisode {}\tAvg Score: {:.2f}\tAvg Score Info: {:.2f}\tEpsilon: {:.2f}\tTime taken: {}'.format(i_episode, np.mean(scores_window), np.mean(scores_window_info), eps, time_taken),
              end="")

        if i_episode % 100 == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(model_dir, str(i_episode) + '_policy.pt'))
            torch.save(agent.target_net.state_dict(), os.path.join(model_dir, str(i_episode) + '_target.pt'))
            scores_save = pd.DataFrame({'Scores': scores})
            scores_save['Scores Info'] = scores_info
            scores_save.to_csv(model_dir + '/' + str(i_episode) + '_' + 'train_scores.csv', sep=',', encoding='utf-8',
                               header=True, index=None)

    time_taken_total = str(timedelta(seconds=(time.time() - start_time_total)))
    print("\nTotal time taken: " + time_taken_total)

    return scores, scores_info


if opt.train:
    scores, scores_info = train(1300)

env.viewer = None
state = stack_frames(None, env.reset(), True)
for j in range(10000):
    env.render(close=False)
    action = agent.act_test(state, eps=0.3)
    next_state, reward, done, _ = env.step(possible_actions[action])
    state = stack_frames(state, next_state, False)
    if done:
        env.reset()
        break
env.render(close=True)