import argparse
import numpy as np
import os
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument('--env-name', type=str, default="cartpole")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--render', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--use_gae', default=False, action='store_true')  # use GAE method in PPO
    parser.add_argument("--use_state_norm", default=False, action='store_true', help="state normalization")
    parser.add_argument("--use_reward_norm", default=False, action='store_true', help="reward normalization")
    parser.add_argument("--use_reward_scaling", default=False, action='store_true', help="reward scaling")
    parser.add_argument("--use_lr_decay", type=lambda x:bool(strtobool(x)), default=False, help="learning rate Decay")

    args = parser.parse_args()

    return args