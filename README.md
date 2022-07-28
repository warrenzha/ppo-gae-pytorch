# SSRL-dynamic-mmwave-mesh
Pytorch implementation of PPO-based reinforcement agent.

## Introduction

This repository provides a PyTorch implementation of Proximal Policy Optimization (PPO) with clipped objective for OpenAI gym environments. It can still be used for complex environments but may require some hyperparameter-tuning or changes in the code.

A concise explaination of PPO algorithm can be found [here](https://stackoverflow.com/questions/46422845/what-is-the-way-to-understand-proximal-policy-optimization-algorithm-in-rl).


## Usage

- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`
- To plot graphs using log files : run `plot_graph.py`
- To save images for gif and make gif using a preTrained network : run `make_gif.py`
- All parameters and hyperparamters to control training / testing / graphs / gifs are in their respective `.py` file
- `PPO_colab.ipynb` combines all the files in a jupyter-notebook
- All the **hyperparameters used for training (preTrained) policies are listed** in the [`README.md` in PPO_preTrained directory](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master/PPO_preTrained)

#### Note :
  - if the environment runs on CPU, use CPU as device for faster training. Box-2d and Roboschool run on CPU and training them on GPU device will be significantly slower because the data will be moved between CPU and GPU often
