import torch.nn as nn

class config_cartpole:
    def __init__(self, seed):
        self.env_name="CartPole-v1"

        # environment hyperparameters
        self.has_continuous_action_space = False
        self.max_ep_len = 400                        # max timesteps in one episode
        self.max_training_timesteps = int(1e5)       # break training loop if timeteps > max_training_timesteps
        self.action_std = None
        self.save_model_freq = int(2e4)              # save model frequency (in num timesteps)

        ## Note : print/log frequencies should be > than max_ep_len

        # PPO hyperparameters
        self.update_timestep = self.max_ep_len * 4  # update policy every n timesteps

        self.K_epochs = 40  # update policy for K epochs
        self.eps_clip = 0.2  # clip parameter for PPO
        self.gamma = 0.99  # discount factor

        self.lr_actor = 0.0003  # learning rate for actor network
        self.lr_critic = 0.001  # learning rate for critic network

        self.random_seed = seed  # set random seed if required (0 = no random seed)

        self.total_test_episodes = 10  # total num of testing episodes


class config_lunarlander:
    def __init__(self, seed):
        self.env_name="LunarLander-v2"

        # environment hyperparameters
        self.has_continuous_action_space = False
        self.max_ep_len = 300                        # max timesteps in one episode
        self.max_training_timesteps = int(1e6)       # break training loop if timeteps > max_training_timesteps
        self.action_std = None
        self.save_model_freq = int(5e4)              # save model frequency (in num timesteps)

        ## Note : print/log frequencies should be > than max_ep_len

        # PPO hyperparameters
        self.update_timestep = self.max_ep_len * 3  # update policy every n timesteps

        self.K_epochs = 30  # update policy for K epochs
        self.eps_clip = 0.2  # clip parameter for PPO
        self.gamma = 0.99  # discount factor

        self.lr_actor = 0.0003  # learning rate for actor network
        self.lr_critic = 0.001  # learning rate for critic network

        self.random_seed = seed  # set random seed if required (0 = no random seed)

        self.total_test_episodes = 10  # total num of testing episodes


class config_bipedalwalker:
    def __init__(self, seed):
        self.env_name="BipedalWalker-v2"

        # environment hyperparameters
        self.has_continuous_action_space = True

        self.max_ep_len = 1500                        # max timesteps in one episode
        self.max_training_timesteps = int(3e6)       # break training loop if timeteps > max_training_timesteps

        self.action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        self.action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        self.min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)
        self.action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

        self.save_model_freq = int(1e5)              # save model frequency (in num timesteps)

        ## Note : print/log frequencies should be > than max_ep_len

        # PPO hyperparameters
        self.update_timestep = self.max_ep_len * 4  # update policy every n timesteps

        self.K_epochs = 80  # update policy for K epochs
        self.eps_clip = 0.2  # clip parameter for PPO
        self.gamma = 0.99  # discount factor

        self.lr_actor = 0.0003  # learning rate for actor network
        self.lr_critic = 0.001  # learning rate for critic network

        self.random_seed = seed  # set random seed if required (0 = no random seed)

        self.total_test_episodes = 10  # total num of testing episodes



def get_config(env_name, seed=0):
    if env_name == 'cartpole':
        return config_cartpole(seed)
    elif env_name == 'lunarlander':
        return config_lunarlander(seed)
    elif env_name == 'bipedalwalker':
        return config_bipedalwalker(seed)