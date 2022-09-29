import argparse
import numpy as np
import os
from distutils.util import strtobool


def parse_args():
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")

    # environment
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument('--env_name', type=str, default="cartpole")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--render', type=lambda x:bool(strtobool(x)), default=False)

    # PPO discrete
    parser.add_argument("--max_train_steps", type=int, default=int(3e6),
                        help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e4,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1e5,
                        help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian",
                        help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=1024,
                        help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4,
                        help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=0.001,
                        help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95,
                        help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=80,
                        help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=lambda x:bool(strtobool(x)), default=True,
                        help="Advantage normalization")
    parser.add_argument("--use_state_norm", type=lambda x:bool(strtobool(x)), default=False,
                        help="State normalization")
    parser.add_argument("--use_reward_norm", type=lambda x:bool(strtobool(x)), default=False,
                        help="Reward normalization")
    parser.add_argument("--use_reward_scaling", type=lambda x:bool(strtobool(x)), default=True,
                        help="Reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Policy entropy")
    parser.add_argument("--use_lr_decay", type=lambda x:bool(strtobool(x)), default=False,
                        help="Learning rate Decay")
    parser.add_argument("--use_grad_clip", type=lambda x:bool(strtobool(x)), default=True,
                        help="Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=lambda x:bool(strtobool(x)), default=True,
                        help="Orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=lambda x:bool(strtobool(x)), default=True,
                        help="Set Adam epsilon=1e-5")

    args = parser.parse_args()

    return args