from collections import defaultdict
import numpy as np
import gymnasium as gym
from copy import deepcopy
import torch
import torch.nn as nn

from agents.q_agent import QAgent
from networks.dqn import DQN
from utils.memory import ReplayMemory, Transition

class DQNAgent(QAgent):

    # Agent methods
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
        with_h_values: bool = True
    ):
        super().init_qlearning_params(
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            action_space=action_space,
            discount_factor=discount_factor
        )
        self.observation_space = observation_space
        self.action_space = action_space
        if type(observation_space) == gym.spaces.discrete.Discrete:
            n_obs = observation_space.n
        else:
            n_obs = len(observation_space)
        # print("n_obs: ", n_obs)
        n_actions = self.action_space.n

        self.policy_net = DQN(n_obs, n_actions)
        self.target_net = deepcopy(self.policy_net)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.with_h_values = with_h_values
        if with_h_values:
            self.belief_policy_net = DQN(n_obs, n_actions)
            self.belief_target_net = deepcopy(self.belief_policy_net)
            self.belief_optimizer = torch.optim.Adam(self.belief_policy_net.parameters(), lr=self.lr)

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
            # we unsqueeze the obs tensor to add a dimension.
            # so if obs shape is (3,) it will become (1, 3)
            # which facillitates using torch.cat to create batch later
            next_obs = torch.tensor(next_obs, device=self.device, dtype=torch.float32).unsqueeze(0)
        
        obs = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)

        action = torch.tensor(action, device=self.device, dtype=torch.int64)
        # if action is 0 dimensional tensor, we need to add a dimension
        if action.dim() == 0:
            action = action.unsqueeze(0)

        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        # if reward is 0 dimensional tensor, we need to add a dimension
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)

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

        # we need to add a dimension to action_batch and reward_batch
        # in order for "gather" call to work later
        action_batch.unsqueeze_(1)
        
        # state_action_values returns a tensor of shape (batch_size, n_actions)
        # so we use the actions chosen by the agent in each batch
        # to select the corresponding Q values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            next_state_values.unsqueeze(1)
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
    def get_q_values_for_obs(self, obs):
        return self.policy_net(torch.tensor(obs, dtype=torch.float32)).detach().numpy()

    def select_action_from_policy(self, obs: tuple[int, int, bool]) -> int:
        """Select an action through the policy network."""
        with torch.no_grad():
            # we use item() below to convert the tensor to a scalar more easily 
            # interpretable by other general code
            return self.policy_net(torch.tensor(obs, dtype=torch.float32)).argmax().item()

    # DQN specific methods
    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path+'.pth')
    
    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net = deepcopy(self.policy_net)

    def generate_q_table(self, env) -> np.ndarray:
        """Generate a Q-table from the policy network."""
        observation_space = self.observation_space
        action_space = self.action_space

        self._observation_type = type(observation_space)
        if self._observation_type == gym.spaces.tuple.Tuple:
            n_obs = tuple(map(lambda x: x.n, observation_space))
            q_shape = n_obs + (action_space.n, )
        elif self._observation_type == gym.spaces.discrete.Discrete:
            n_obs = observation_space.n
            #in this case, shape should be something like (n_obs, action_space.n)
            q_shape = (n_obs, action_space.n)

        # use env to generate observation space
        print("q_shape", q_shape)
        q_table = np.zeros(q_shape)

        # for state in range(q_shape[0]):
        #     for action in range(q_shape[1]):
        #         q_table[state, action] = self.policy_net(torch.tensor(state, dtype=torch.float32)).max().item()
        return q_table