# This file contains the argument parser for the main.py file.
# All possible arguments and default values are defined here.
import argparse
import json

def init_parser() -> argparse.ArgumentParser:
    """Initializes the argument parser with all the possible arguments and their default values."""

    parser = argparse.ArgumentParser(
        description='Test XRL techniques on different environments.')

    # Defining command line arguments

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

    # Defining defaults for some arguments

    parser.set_defaults(should_print=True)
    parser.set_defaults(deep=False)

    return parser

def parse_args(sys_arguments) -> dict:
    parser = init_parser()
    """Returns the parsed arguments from the command line. If the -f option is used,
    then the arguments will be parsed from a JSON file instead of the command line."""
    parsing_result = parser.parse_args(sys_arguments)

    print("parsing_result: ", parsing_result)

    if parsing_result.file_path:
        # parse arguments from JSON file
        file_path = parsing_result.file_path
        print(f"Loading program parameters from file: {file_path}")
        with open(file_path, 'r') as f:
            parsing_result = dict(json.load(f))
        parsing_result = add_missing_params(parsing_result)
    else:
        # maintain original argparse.Namespace object
        # but convert it to a dictionary
        parsing_result = vars(parsing_result)
        # also save params passed
        save_params(parsing_result)
    return parsing_result

def save_params(arg_dictionary: dict) -> str:
    """Saves the parameters used in the experiment to JSON file. Used only for logging purposes if
    user did not pass the -f option. Returns the path to the saved file."""
    save_file = "results/params.json"
    with open(save_file, 'w') as f:
        json.dump(arg_dictionary, f, indent=4)
    return save_file

def add_missing_params(arg_dictionary: dict) -> dict:
    """Adds missing parameters to the dictionary. Used to avoid errors when accessing
    parameters that were not passed by the user. Returns the original dictionary with the added
    parameters."""
    dict_copy = arg_dictionary.copy()
    missing_params = {
        'technique': 'hvalues',
        'environment': 'blackjack',
        'num_episodes': 100_000,
        'load_path': None,
        'file_path': None,
        'should_print': True,
        'deep': False,
        'agent': {}
    }
    for param in missing_params:
        if param not in arg_dictionary:
            dict_copy[param] = missing_params[param]
    return dict_copy