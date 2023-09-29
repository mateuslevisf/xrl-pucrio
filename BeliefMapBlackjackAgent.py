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
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(action_space.n))
        # the matrix H/the h values correspond to the "belief map" from the WDYTWH paper
        # we will learn this map of discounted expected states concurrently with the Q values
        self.h_values = defaultdict(lambda: np.zeros(action_space.n))

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
        print("qvalue before", self.q_values[obs][action])
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        expected_reward = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * expected_reward
        )
        print("qvalue after", self.q_values[obs][action])

        # Updating H values
        print("hvalue before", self.h_values[obs][action])
        intent_update = defaultdict(lambda: np.zeros(self.action_space.n))
        intent_update[obs][action] = 1
        future_h_value = (not terminated) * np.max(self.h_values[next_obs][np.argmax(self.q_values[next_obs])])
        expected_intent = (
            intent_update + self.discount_factor * future_h_value - self.h_values[obs][action]
        )
        self.h_values[obs][action] = (
            self.h_values[obs][action] + self.lr * expected_intent
        )
        print("hvalue after", self.h_values[obs][action])

        self.training_error.append(expected_reward)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)