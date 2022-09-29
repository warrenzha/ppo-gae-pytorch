import numpy as np
import torch
import torch.nn as nn


################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def orthogonal_init(layer, gain=1.0):
    """Orthogonal initialization."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


# Another version of replay buffer
class ReplayBuffer:
    def __init__(self, args):
        self.capacity = args.batch_size
        self.has_continuous_action_space = args.has_continuous_action_space

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        # obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.s = np.empty((args.batch_size, args.state_dim), dtype=np.float32)
        if args.has_continuous_action_space:
            self.a = np.empty((args.batch_size, args.action_dim), dtype=np.float32)
            self.a_logprob = np.empty((args.batch_size, args.action_dim), dtype=np.float32)
        else:
            self.a = np.empty((args.batch_size, 1), dtype=np.float32)
            self.a_logprob = np.empty((args.batch_size, 1), dtype=np.float32)
        self.r = np.empty((args.batch_size, 1), dtype=np.float32)
        self.s_ = np.empty((args.batch_size, args.state_dim), dtype=np.float32)
        self.dw = np.empty((args.batch_size, 1), dtype=np.float32)
        self.done = np.empty((args.batch_size, 1), dtype=np.float32)

        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        np.copyto(self.s[self.count], s)
        np.copyto(self.a[self.count], a)
        np.copyto(self.a_logprob[self.count], a_logprob)
        np.copyto(self.r[self.count], r)
        np.copyto(self.s_[self.count], s_)
        np.copyto(self.dw[self.count], dw)
        np.copyto(self.done[self.count], done)
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.as_tensor(self.s).float().to(device)
        if self.has_continuous_action_space:
            a = torch.as_tensor(self.a).float().to(device)
        else:
            a = torch.as_tensor(self.a).long().to(device)   # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.as_tensor(self.a_logprob).float().to(device)
        r = torch.as_tensor(self.r).float().to(device)
        s_ = torch.as_tensor(self.s_).float().to(device)
        dw = torch.as_tensor(self.dw).float().to(device)
        done = torch.as_tensor(self.done).float().to(device)

        return s, a, a_logprob, r, s_, dw, done


class RolloutBuffer:
    def __init__(self, args):
        self.has_continuous_action_space = args.has_continuous_action_space

        self.s = np.zeros((args.batch_size, args.state_dim))
        if args.has_continuous_action_space:
            self.a = np.zeros((args.batch_size, args.action_dim))
            self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        else:
            self.a = np.zeros((args.batch_size, 1))
            self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        if self.has_continuous_action_space:
            a = torch.tensor(self.a, dtype=torch.float)
        else:
            a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)