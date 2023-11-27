# Adapted from https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/#sphx-glr-tutorials-training-agents-blackjack-tutorial-py
# Adaptations: class was made more general by taking out use of global variables in exchange for parameters such as num_actions
from collections import defaultdict
import numpy as np
import gymnasium as gym

class BeliefMapBlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        action_space: gym.Space,
        observation_space: gym.Space,
        discount_factor: float = 0.95
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            num_actions: The number of available actions in the environment
            action_space: The action space of the environment
            observation_space: The observation space of the environment
            discount_factor: The discount factor for computing the Q-value
        """
        n_obs = tuple(map(lambda x: x.n, observation_space))
        q_shape = n_obs + (action_space.n, )
        self.q_values = np.zeros(shape = q_shape)
        # the matrix H/the h values correspond to the "belief map" from the WDYTWH paper
        # we will learn this map of discounted expected states concurrently with the Q values
        self.h_values = np.zeros(self.q_values.shape)

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.action_space = action_space

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):  
        """Updates the Q-value and H-value of an action."""
        # Updating Q values

        # argmax_a is the action that maximizes the q value of the next state (not the current one)
        argmax_a = np.argmax(self.q_values[next_obs])
        future_q_value = (not terminated) * argmax_a
        expected_reward = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )
        self.q_values[obs + (action,)] += self.lr * expected_reward

        # Updating H values

        # below code is closer to the WDYTWH paper but it doesn't really make sense to me
        # h_update = np.zeros(self.h_values.shape)
        # h_update[obs + (action, )] += 1

        h_update = 1
        future_h_value = (not terminated) * self.h_values[next_obs + (argmax_a, )]
        expected_h = h_update + self.discount_factor * future_h_value- self.h_values[obs + (action, )]
        self.h_values[obs + (action,)] += self.lr * expected_h

        self.training_error.append(expected_reward)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)