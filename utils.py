import numpy as np
import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    PPO requires orthogonal initialization.

    Args:
        layer:
        std:
        bias_const:

    Returns:
        layer:
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


