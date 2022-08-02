import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gym
# import roboschool
# import pybullet

from arguments import parse_args
from config import get_config
from model import PPO



#################################### Testing ###################################
def test(args):
    print("============================================================================================")

    ################## hyperparameters ##################
    config = get_config(args.env_name, args.seed)

    env_name = config.env_name
    has_continuous_action_space = config.has_continuous_action_space
    max_ep_len = config.max_ep_len
    action_std = config.action_std

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    # env_name = "HalfCheetah-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1000  # max timesteps in one episode
    # action_std = 0.1  # set same std for action distribution which was used while saving

    # env_name = "RoboschoolWalker2d-v1"
    # has_continuous_action_space = True
    # max_ep_len = 1000           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    render = args.render          # render environment on screen
    frame_delay = 0               # if required; add delay b/w frames

    total_test_episodes = config.total_test_episodes    # total num of testing episodes

    K_epochs = config.K_epochs              # update policy for K epochs
    eps_clip = config.eps_clip              # clip parameter for PPO
    gamma = config.gamma                    # discount factor

    lr_actor = config.lr_actor              # learning rate for actor
    lr_critic = config.lr_critic            # learning rate for critic

    #####################################################

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = config.random_seed             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = args.pretrained      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    args = parse_args()
    print(args)

    test(args)