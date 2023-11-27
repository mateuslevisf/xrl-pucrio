from collections import defaultdict
import numpy as np
import gymnasium as gym
from copy import deepcopy
import torch

from blackjack.q_agent import QAgent
from networks.dqn import DQN

class DQNAgent(QAgent):
    def __init__(self, **kwargs):
        super.__init__(kwargs)

        n_obs = tuple(map(lambda x: x.n, self.observation_space))
        n_actions = self.action_space.n

        self.policy_net = DQN(n_obs, n_actions)
        self.target_net = deepcopy(self.policy_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
    
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        return super.get_action(obs)

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):  
        super.update(obs, action, reward, terminated, next_obs)

    def decay_epsilon(self):
        return super().decay_epsilon()
