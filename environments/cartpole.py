from environments.env_instance import EnvironmentInstance
from utils.wrappers import DiscretizedObservationWrapper
import numpy as np

class CartpoleEnvironment(EnvironmentInstance):
    def __init__(self, deep=False, **kwargs):
        super().__init__("CartPole-v1", **kwargs)

        # we want to convert when we are not using deep learning
        convert = not deep

        # we must first wrap the environment with the DiscretizedObservationWrapper
        # in order to discretize the observation space; cartpole has a continuous 
        # observation space, which is not supported by the Q-learning algorithm
        self._instance = DiscretizedObservationWrapper(self._instance, 
            n_bins=np.array([3, 3, 6, 3]),
            low=[-2.4, -2.5, -np.radians(12), -1],
            high=[2.4, 2.5, np.radians(12), 1],
            convert=convert
        )
        self._observation_space = self._instance.observation_space
        self._action_space = self._instance.action_space

    def loop(self, *kargs ,**kwargs):
        return super().loop(*kargs, **kwargs, already_wrapped=True)