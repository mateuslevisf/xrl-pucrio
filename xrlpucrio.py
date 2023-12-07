# General imports
import sys
import os
import shutil

# Importing util functions
from utils.arguments_parser import parse_args
from utils.log import log, set_should_print, show_running_info

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
    folder = 'results'
    for root, _, files in os.walk(folder):
        for f in files:
            # check if file is .gitkeep
            if f != '.gitkeep':
                os.unlink(os.path.join(root, f))

    args = parse_args(sys.argv[1:])

    set_should_print(args['should_print'])
    show_running_info(args)
    
    environment = args['environment']
    technique = args['technique']

    deep = args['deep']
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
    epsilon_decay = initial_epsilon / (num_episodes / 2)
    learning_rate = args['agent']['learning_rate'] if args['agent']['learning_rate'] is not None else 0.01
    discount_factor = args['agent']['discount_factor'] if args['agent']['discount_factor'] is not None else 1
    # if environment == 'cartpole':
    #     # 0.0001 was bad
    #     # 0.1 was bad
    #     # 0.01 showed best results until now but 100_000 episodes was too little to stabilize
    #     # testing with 1mil -> two peaks at 90 in eval but still not stable
    #     learning_rate = 0.01
    #     discount_factor = 1
    num_episodes = args['num_episodes']
    initial_epsilon = args['agent']['initial_epsilon'] if args['agent']['initial_epsilon'] is not None else 1
    final_epsilon = args['agent']['final_epsilon'] if args['agent']['final_epsilon'] is not None else 0.01
    hidden_dim = args['agent']['hidden_dim'] if args['agent']['hidden_dim'] is not None else 64

    params = {
        "learning_rate": learning_rate,
        "initial_epsilon": initial_epsilon,
        "epsilon_decay": epsilon_decay,
        "final_epsilon": final_epsilon,
        "action_space": env_action_space,
        "observation_space": env_observation_space,
        "discount_factor": discount_factor,
        "hidden_dim": hidden_dim
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

    agent.save('results/agent')

    if technique == 'viper':
        decision_tree = train_viper(agent, env, 100, 200)
        decision_tree.save('results/')

    env.close()

if __name__ == "__main__":
    main()