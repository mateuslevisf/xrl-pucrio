# This file contains the argument parser for the main.py file.
# All possible arguments and default values are defined here.
import argparse
import json

parser = argparse.ArgumentParser(
    description='Test XRL techniques on different environments.')

parser.add_argument('-t', '--technique', dest='technique', type=str, help='The XRL technique to be tested.', 
    default='hvalues', choices=['hvalues', 'viper'])
parser.add_argument('-e', '--environment', dest='environment', type=str, 
    help='The environment to test the technique on.', default='blackjack',
    choices=['blackjack', 'cartpole'])
parser.add_argument('-n', '--num_episodes', dest='num_episodes', type=int,
    help='The number of episodes to run the technique on.', default=100_000)
parser.add_argument('-l', '--load', dest='load_path', 
    help='Load a saved model by passing path.')
parser.add_argument('-f', '--file', dest='file_path', 
    help='Pass program parameters from a JSON file instead of command line. \
    Using this option will override any command line arguments.')

# parser.add_argument('-d', '-deep', dest='deep', action='store_true', 
#     help='Enable deep learning if available for chosen technique and environment.')
parser.add_argument('--noprint', dest='should_print', action='store_false', help='Disable log printing.')

parser.set_defaults(should_print=True)
parser.set_defaults(deep=False)

def parse_args() -> dict:
    """Returns the parsed arguments from the command line. If the -f option is used,
    then the arguments will be parsed from a JSON file instead of the command line."""
    parsing_result = parser.parse_args()

    print("parsing_result: ", parsing_result)

    if parsing_result.file_path:
        # parse arguments from JSON file
        file_path = parsing_result.file_path
        print(f"Loading program parameters from file: {file_path}")
        with open(file_path, 'r') as f:
            parsing_result = dict(json.load(f))

        print("parsing_result: ", parsing_result)
    else:
        # maintain original argparse.Namespace object
        # but convert it to a dictionary
        parsing_result = vars(parsing_result)
    return parsing_result