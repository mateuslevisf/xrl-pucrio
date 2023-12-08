import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from environments.env_instance import EnvironmentInstance
from utils.wrappers import DiscretizedObservationWrapper
from utils.plots import plot_table_cartpole

import torch


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

    def generate_plots(self, evaluation_results, agent, deep=False, **kwargs):
        if agent is None:
            raise("Generate plots function for Cartpole environment requires agent to be passed.")
        super().generate_plots(evaluation_results, **kwargs)

        q_data=None
        h_data=None
        if not deep:
            q_data = agent.q_values
            h_data = agent.h_values
        # else:
        #     q_data = agent.generate_q_table(self._instance)

        if q_data is not None:
            plot_table_cartpole(q_data, title="Q-Values")
            plt.savefig("results/images/cartpole_q_values.png")
            
        if h_data is not None:
            plot_table_cartpole(h_data, title="H-Values")
            plt.savefig("results/images/cartpole_h_values.png")