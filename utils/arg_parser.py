import argparse

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