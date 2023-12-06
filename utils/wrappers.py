import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Discrete, Tuple
import numpy as np
from copy import deepcopy
from functools import reduce

# taken from https://github.com/hmhyau/rl-intention/blob/main/wrappers.py

class DiscretizedObservationWrapper(ObservationWrapper):
    """
        Discretize the observation space by rounding values to the nearest bin.
        Necessary for Q-learning and tabular methods to work with environments
        with continuous observation spaces.
    """
    def __init__(self, env, n_bins=None, low=None, high=None, convert=True):
        super().__init__(env)
        self.convert = convert
        assert isinstance(env.observation_space, Box)
        self.obs_shape = self.observation_space.shape
        assert n_bins.shape == self.obs_shape

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        low = np.array(low)
        high = np.array(high)

        self.n_bins = n_bins
        self.disc_bins = [np.linspace(l, h, bin+1) for l, h, bin in 
                          zip(low.flatten(), high.flatten(), n_bins)]
        self.disc_r = [np.linspace(l, h, bin+1)[1:-1] for l, h, bin in
                       zip(low.flatten(), high.flatten(), n_bins)]

        # preserve original observation space info
        self.orig_observation_space = deepcopy(self.observation_space)
        if convert:
            self.observation_space = Discrete(np.prod(self.n_bins))
        else:
            self.observation_space = Tuple([Discrete(x) for x in self.n_bins])

    def _convert_to_int(self, digits):
        out = 0
        for i in reversed(range(len(self.disc_bins))):
            if i == 0:
                out += digits[-(i+1)]
            else:
                out += reduce(lambda x, y: x*y, self.n_bins[-i:]) * digits[-(i+1)]

        return out

    def observation(self, observation):
        digits = [np.digitize(x=x, bins=bins) for x, bins in zip(observation.flatten(), self.disc_r)]
        # print(digits, self.disc_r)
        if self.convert:
            return self._convert_to_int(digits)
        
        return np.array(digits).astype(np.float32)