import sys
import argparse

from utils.log import log, set_should_print, show_running_info
from environments.blackjack.blackjack import BlackjackEnvironment

from environments.blackjack.q_agent import QAgent
from environments.blackjack.dqn_agent import DQNAgent

parser = argparse.ArgumentParser(
    description='Test XRL techniques on different environments.')
parser.add_argument('-t', '--technique', dest='technique', type=str, help='The XRL technique to be tested.', 
    default='hvalues', choices=['hvalues', 'viper'])
parser.add_argument('-e', '--environment', dest='environment', type=str, 
    help='The environment to test the technique on.', default='blackjack',
    choices=['blackjack', 'cartpole'])
parser.add_argument('-n', '--num_episodes', dest='num_episodes', type=int,
    help='The number of episodes to run the technique on.', default=100_000)

parser.add_argument('-d', '-deep', dest='deep', action='store_true', 
    help='Enable deep learning if available for chosen technique and environment.')
parser.add_argument('--noprint', dest='should_print', action='store_false', help='Disable log printing.')

parser.set_defaults(should_print=True)
parser.set_defaults(deep=False)

def main():
    args = parser.parse_args()
    set_should_print(args.should_print)
    show_running_info(vars(args))
    
    environment = args.environment
    technique = args.technique

    if environment == 'blackjack' and technique == 'hvalues':
        env = BlackjackEnvironment(sab=True)
    else:
        log("Invalid combination of environment and technique.")
        sys.exit(1)

    env_action_space = env.get_action_space()
    env_observation_space = env.get_observation_space()

    # Agent hyperparameters definitions
    learning_rate = 0.01
    num_episodes = args.num_episodes
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (num_episodes / 2)
    final_epsilon = 0.1

    params = {
        "learning_rate": learning_rate,
        "initial_epsilon": initial_epsilon,
        "epsilon_decay": epsilon_decay,
        "final_epsilon": final_epsilon,
        "action_space": env_action_space,
        "observation_space": env_observation_space,
    }
    
    if not args.deep:
        log("Using Q-learning agent")
        agent = QAgent(**params)
    else:
        log("Using DQN agent")
        agent = DQNAgent(**params)

    # Execution setup
    evaluation_interval = 1000
    evaluation_duration = num_episodes//100

    evaluation_results = env.loop(agent, num_episodes, evaluation_interval, evaluation_duration)

    env.generate_plots(agent, evaluation_results, args.deep)

    env.close()

if __name__ == "__main__":
    main()