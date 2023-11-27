import random
import math
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from lbforaging.foraging.agent import Agent
from lbforaging.agents.networks.dqn import DQN
from lbforaging.foraging.environment import Action, ForagingEnv as Env
from lbforaging.agents.helpers import ReplayMemory, Transition

# Based on a simple DQN implementation from PyTorch, adapted to Agent class
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html