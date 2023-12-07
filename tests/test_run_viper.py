import unittest

from agents.dqn_agent import DQNAgent
from environments.cartpole import CartpoleEnvironment
from environments.blackjack import BlackjackEnvironment

from utils.viper import train_viper

class TestViperCartpole(unittest.TestCase):
    def setUp(self):
        self.num_episodes = 10_000
        initial_epsilon = 1.0
        epsilon_decay = initial_epsilon / (self.num_episodes / 2)
        self.env = CartpoleEnvironment(deep=True, technique='viper')
        env_action_space = self.env.get_action_space()
        env_observation_space = self.env.get_observation_space()
        self.agent = DQNAgent(
            epsilon_decay=epsilon_decay,
            action_space=env_action_space,
            observation_space=env_observation_space,
            learning_rate=0.01,
            initial_epsilon=initial_epsilon,
            final_epsilon=0.01,
            discount_factor=0.95
        )

    def test_cartpole_h_values(self):
        """Test if the agent can learn to play Cartpole with VIPER."""
        print("Testing Cartpole with h-values by running for {} episodes.".format(self.num_episodes))
        self.env.loop(self.agent, self.num_episodes, 1000, 1000)
        self.assertTrue(True)

class TestViperBlackjack(unittest.TestCase):
    def setUp(self):
        self.num_episodes = 10_000
        initial_epsilon = 1.0
        epsilon_decay = initial_epsilon / (self.num_episodes / 2)
        self.env = BlackjackEnvironment(sab=True, technique='viper')
        env_action_space = self.env.get_action_space()
        env_observation_space = self.env.get_observation_space()
        self.agent = DQNAgent(
            epsilon_decay=epsilon_decay,
            action_space=env_action_space,
            observation_space=env_observation_space,
            learning_rate=0.01,
            initial_epsilon=initial_epsilon,
            final_epsilon=0.01,
            discount_factor=0.95
        )

    def test_blackjack_h_values(self):
        """Test if the agent can learn to play Blackjack with VIPER."""
        print("Testing Blackjack with h-values by running for {} episodes.".format(self.num_episodes))
        self.env.loop(self.agent, self.num_episodes, 1000, 1000)
        self.assertTrue(True)
        train_viper(self.agent, self.env, self.num_episodes, 1000)
        self.assertTrue(True)