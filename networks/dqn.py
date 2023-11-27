import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # Standard DQN architecture:
        # DQN takes the observations as input and outputs the Q-values for each action in the given state
        # 1 input layer, 1 hidden layer with 128 neurons and 1 output layer with n_actions neurons
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # Called with either one element to determine next action, or a batch
        # during optimization. Returns Q-values for each action;
        # Q_a (o_t) where o_t is the observation at time t
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
