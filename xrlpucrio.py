# General imports
import sys
import os
import shutil

# Importing util functions
from utils.arguments_parser import parser
from utils.log import log, set_should_print, show_running_info
from utils.parameters import save_params

# Importing own environments
from environments.blackjack import BlackjackEnvironment
from environments.cartpole import CartpoleEnvironment

# Importing own agents
from agents.q_agent import QAgent
from agents.dqn_agent import DQNAgent

# Importing viper functions
from utils.viper import train_viper

def main():
    # first we clean up the images folder
    folder = 'images'
    for root, _, files in os.walk(folder):
        for f in files:
            os.unlink(os.path.join(root, f))

    args = parser.parse_args()
            
    save_params(args)

    set_should_print(args.should_print)
    show_running_info(vars(args))
    
    environment = args.environment
    technique = args.technique

    deep = False
    if technique == 'viper':
        deep = True

    if environment == 'blackjack':
        env = BlackjackEnvironment(sab=True, technique=technique, deep=deep)
    elif environment == 'cartpole':
        env = CartpoleEnvironment(deep=deep, technique=technique)
    else:
        log("Invalid combination of environment and technique.")
        sys.exit(1)
    
    env_action_space = env.get_action_space()
    env_observation_space = env.get_observation_space()

    # Agent hyperparameters definitions
    learning_rate = 0.01
    discount_factor = 0.95
    if environment == 'cartpole':
        # 0.0001 was bad
        # 0.1 was bad
        # 0.01 showed best results until now but 100_000 episodes was too little to stabilize
        # testing with 1mil -> two peaks at 90 in eval but still not stable
        learning_rate = 0.01
        discount_factor = 1
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
        "discount_factor": discount_factor,
    }
    
    if not deep:
        log("Using Q-learning agent")
        agent = QAgent(**params)
    else:
        log("Using DQN agent")
        agent = DQNAgent(**params)

    # Execution setup
    # evaluation_interval = 1000
    evaluation_interval = num_episodes//50
    evaluation_duration = 1000

    evaluation_results = env.loop(agent, num_episodes, evaluation_interval, evaluation_duration)

    env.generate_plots(evaluation_results, agent=agent, deep=deep)

    if technique == 'viper':
        decision_tree = train_viper(agent, env, 100, 200)

    env.close()

if __name__ == "__main__":
    main()