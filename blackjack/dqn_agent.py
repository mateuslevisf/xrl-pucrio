from collections import defaultdict
import numpy as np
import gymnasium as gym
from copy import deepcopy
import torch

from blackjack.q_agent import QAgent
from networks.dqn import DQN

class DQNAgent(QAgent):
    def __init__(self, 
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        action_space: gym.Space,
        observation_space: gym.Space,
        discount_factor: float = 0.95,
        batch_size: int = 128,
        target_update: float = 0.005,
    ):
        super().__init__(learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            action_space=action_space,
            observation_space=observation_space,
            discount_factor=discount_factor,
            with_h_values=False)

        n_obs = len(observation_space)
        n_actions = self.action_space.n

        self.policy_net = DQN(n_obs, n_actions)
        self.target_net = deepcopy(self.policy_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.tau = target_update
        self.batch_size = batch_size

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):  
        super().update(obs, action, reward, terminated, next_obs)

    # Override the QAgent's select_action_from_policy method
    def select_action_from_policy(self, obs: tuple[int, int, bool]) -> int:
        """Select an action through the policy network."""
        with torch.no_grad():
            print("selected:", self.policy_net(torch.tensor(obs, dtype=torch.float32)).argmax().item())
            return self.policy_net(torch.tensor(obs, dtype=torch.float32)).argmax().item()
