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


    args = parser.parse_args()

    return args