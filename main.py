import numpy as np
import gymnasium as gym

should_print = True

def log(message):
    if should_print:
        print(message)

# Make Blackjack environment.
# 'sab' parameter defines environment following Sutton and Barton's book rules.
env = gym.make("Blackjack-v1", sab=True)

# We reset the environment. Done will be used to check if the game is over later;
done = False
observation, info = env.reset()

# Observation follows the format: (player's current sum, dealer's face-up card, boolean whether the player has an usable ace)
# Usable ace = an ace that can be used as 11 without the player going bust.
log("Initial observation: {}".format(observation))




