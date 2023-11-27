# External dependencies
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Internal dependencies
from blackjack.q_agent import BlackjackQAgent
from plot_utils import create_grids, create_plots, plot_table_blackjack, plot_error
from helpers import log

# Adapted from https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/#sphx-glr-tutorials-training-agents-blackjack-tutorial-py

def run_blackjack(should_print=False):
    global log
    # Make Blackjack environment.
    # 'sab' parameter defines environment following Sutton and Barton's book rules.
    env = gym.make("Blackjack-v1", sab=True)

    # We reset the environment. Done will be used to check if the game is over later;
    done = False
    observation, info = env.reset()

    # Observation follows the format: (player's current sum, dealer's face-up card, boolean whether the player has an usable ace)
    # Usable ace = an ace that can be used as 11 without the player going bust.
    log("Initial observation: {}".format(observation))

    # Hyperparameters definitions
    learning_rate = 0.01
    num_episodes = 100_000
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (num_episodes / 2)
    final_epsilon = 0.1

    agent = BlackjackQAgent(learning_rate, initial_epsilon, epsilon_decay, final_epsilon, env.action_space, env.observation_space)

    # Training loop
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    for episode in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Update agent
            agent.update(obs, action, reward, terminated, next_obs)

            # Update environment status (done or not) and observation
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    # plot_error(env, agent)
    # plt.show()

    # # state values & policy with usable ace (ace counts as 11)
    # value_grid, policy_grid = create_grids(agent, usable_ace=True)
    # fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
    # plt.show()

    # # state values & policy without usable ace (ace counts as 1)
    # value_grid, policy_grid = create_grids(agent, usable_ace=False)
    # fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
    # plt.show()

    table_cmap = sns.diverging_palette(10, 240, n=128)
    fig3 = plot_table_blackjack(agent.q_values, center = 0, cmap=table_cmap, title="Q-Values")
    fig4 = plot_table_blackjack(agent.h_values, center = 0, cmap=table_cmap, title="H-Values")
    plt.show()

    env.close()