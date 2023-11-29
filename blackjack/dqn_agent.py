from collections import defaultdict
import numpy as np
import gymnasium as gym
from copy import deepcopy
import torch
import torch.nn as nn

from blackjack.q_agent import QAgent
from networks.dqn import DQN
from utils.memory import ReplayMemory, Transition

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
        self.memory = ReplayMemory(10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):  
        if terminated == True:
            next_obs = None
        else:
            next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32)
        
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32)
        action = torch.tensor(action, device=self.device, dtype=torch.int64)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)

        self.memory.push(obs, action, next_obs, reward)

        self.optimize_model()

    def optimize_model(self):
        """Perform a single step of the optimization (on the policy and target network)"""
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Update the target network
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = self.tau * policy_net_state_dict[key] + (1 - self.tau) * target_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)


    # Override the QAgent's select_action_from_policy method
    def select_action_from_policy(self, obs: tuple[int, int, bool]) -> int:
        """Select an action through the policy network."""
        with torch.no_grad():
            print("selected:", self.policy_net(torch.tensor(obs, dtype=torch.float32)).argmax().item())
            return self.policy_net(torch.tensor(obs, dtype=torch.float32)).argmax().item()
