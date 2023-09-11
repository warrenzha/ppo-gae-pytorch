"""
@author: Wenyuan Zhao

Self-supervised Reinforcement Learning
"""


import os
from datetime import datetime

import numpy as np
import gym
import torch
from agent.ppo_discrete import make_ppo_discrete
from agent.ppo_continous import make_ppo_continous

from arguments import parse_args
from env.config import get_config
import utils as utils



def evaluate_policy(args, env, agent, eval_seed, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset(seed=eval_seed)
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.has_continuous_action_space:
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
            else:
                action = a
            s_, r, done, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def train(args, env_name, seed, number=1):
    config = get_config(args.env_name, args.seed)

    env = gym.make(env_name)
    env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment

    # Set random seed
    env.action_space.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # state space dimension
    args.state_dim = env.observation_space.shape[0]

    args.has_continuous_action_space = config.has_continuous_action_space
    # action space dimension
    if args.has_continuous_action_space:
        args.action_dim = env.action_space.shape[0]
        args.max_action = float(env.action_space.high[0])
    else:
        args.action_dim = env.action_space.n

    args.max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode
    print_freq = args.max_episode_steps * 5  # print avg reward in the interval (in num timesteps)
    log_freq = args.max_episode_steps * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = args.save_freq  # save model frequency (in num timesteps)

    #### log files for multiple runs ###################
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #### get number of log files in log directory ######
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    print("============================================================================================")
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = args.pretrained  #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path_actor = directory + "PPO_Actor_{}_{}_{}.pth".format(env_name, seed, run_num_pretrained)
    checkpoint_path_critic = directory + "PPO_Critic_{}_{}_{}.pth".format(env_name, seed, run_num_pretrained)
    print("save actor checkpoint path : " + checkpoint_path_actor)
    print("save critic checkpoint path : " + checkpoint_path_critic)
    print("============================================================================================")
    #####################################################

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    #### Prepare agent and buffer ##############################
    replay_buffer = utils.ReplayBuffer(args)
    if args.has_continuous_action_space:
        agent = make_ppo_continous(args)
    else:
        agent = make_ppo_discrete(args)
    
    state_norm = utils.Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = utils.Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = utils.RewardScaling(shape=1, gamma=args.gamma)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    ################### training #########################
    while total_steps < args.max_train_steps:
        s, info = env.reset(seed=seed)
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()

        episode_steps = 0
        current_ep_reward = 0
        done = False

        while not done:
            episode_steps += 1

            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability
            if args.has_continuous_action_space:
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
            else:
                action = a
            s_, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            current_ep_reward += r
            time_step += 1

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving actor model at : " + checkpoint_path_actor)
                print("saving critic model at : " + checkpoint_path_critic)
                agent.save(checkpoint_path_actor, checkpoint_path_critic)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parse_args()
    print(args)

    config = get_config(args.env_name, args.seed)

    train(
        args=args,
        env_name=config.env_name,
        seed=args.seed,
    )