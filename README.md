# SSRL-dynamic-mmwave-mesh
_Wenyuan Zhao_

Pytorch implementation of PPO-based reinforcement agent.

## Introduction

This repository provides a PyTorch implementation of Proximal Policy Optimization (PPO) with clipped objective for OpenAI gym environments. It can still be used for complex environments but may require some hyperparameter-tuning or changes in the code.

A concise explaination of PPO algorithm can be found [here](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl).


## Usage

- To train a new network : example - train PPO on CartPole domain   
`python train.py --env-name cartpole --seed 0 --pretrained 0`
- To test a preTrained network : example - test PPO on CartPole domain   
`python test.py --env-name cartpole --seed 0 --pretrained 0`
- All parameters and hyperparamters to control training / testing are in `config.py` file.

We provide a pre-trained model that can be used for evaluation. 
Call `python test.py --env-name cartpole --pretrained 1`

#### Note :
  - Device is set to GPU as default if CUDA is available.
  - if the environment runs on CPU, use CPU as device for faster training. Box-2d and Roboschool run on CPU and training them on GPU device will be significantly slower because the data will be moved between CPU and GPU often.
