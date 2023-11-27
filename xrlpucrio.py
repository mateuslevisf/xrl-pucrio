import sys
import argparse

from utils import log, set_should_print
from blackjack.blackjack import run_blackjack

parser = argparse.ArgumentParser(
    description='Test XRL techniques on different environments.')
parser.add_argument('-t', '--technique', dest='technique', type=str, help='The XRL technique to be tested.', 
    default='hvalues', choices=['hvalues', 'viper'])
parser.add_argument('-e', '--environment', dest='environment', type=str, 
    help='The environment to test the technique on.', default='blackjack',
    choices=['blackjack', 'cartpole'])
parser.add_argument('--noprint', dest='should_print', action='store_false', help='Disable log printing.')
parser.set_defaults(should_print=True)

def main():
    args = parser.parse_args()
    set_should_print(args.should_print)
    log("should_print: {}".format(args.should_print))
    
    environment = args.environment
    technique = args.technique

    if environment == 'blackjack' and technique == 'hvalues':
        run_blackjack()
    else:
        log("Invalid combination of environment and technique.")
        sys.exit(1)

if __name__ == "__main__":
    main()