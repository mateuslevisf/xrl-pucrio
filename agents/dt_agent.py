from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

from agents.agent import Agent

def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    obss_train = obss[idx[:n_train]]
    acts_train = acts[idx[:n_train]]
    obss_test = obss[idx[n_train:]]
    acts_test = acts[idx[n_train:]]
    return obss_train, acts_train, obss_test, acts_test

def accuracy(agent, observations, actions):
    return np.mean(agent.get_action(obs) == act for obs, act in zip(observations, actions))

class DTAgent(Agent):
    def __init__(self, max_depth=5):
        super().__init__()
        self._max_depth = max_depth

    def fit(self, observations, actions):
        self._dt = DecisionTreeClassifier(max_depth=self._max_depth)
        self._dt.fit(observations, actions)

    def train(self, observations, actions, train_fraction=0.8) -> dict:
        """Trains the agent on the given observations and actions, and returns a dictionary of training info."""
        obss_train, acts_train, obss_test, acts_test = split_train_test(observations, actions, train_fraction)
        self.fit(observations, actions)
        info = {
            'train_accuracy': accuracy(self, obss_train, acts_train),
            'test_accuracy': accuracy(self, obss_test, acts_test),
            'node_count': self._dt.tree_.node_count,
        }
        return info
        
    def get_action(self, obs, eval: bool = False) -> int:
        return self._dt.predict([obs])[0]