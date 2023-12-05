import os
import pickle as pk
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from agents.agent import Agent

def accuracy(policy, obss, acts):
    return np.mean(acts == policy.predict(obss))

def split_train_test(obss, acts, train_frac):
    n_train = int(train_frac * len(obss))
    idx = np.arange(len(obss))
    np.random.shuffle(idx)
    # separate into train and test
    train_indexes = idx[:n_train]
    test_indexes = idx[n_train:]
    obss_train = [obss[i] for i in train_indexes]
    acts_train = [acts[i] for i in train_indexes]
    obss_test = [obss[i] for i in test_indexes]
    acts_test = [acts[i] for i in test_indexes]
    return obss_train, acts_train, obss_test, acts_test

def save_dt_policy(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    f = open(dirname + '/' + fname, 'wb')
    pk.dump(dt_policy, f)
    f.close()

def save_dt_policy_viz(dt_policy, dirname, fname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    export_graphviz(dt_policy.tree, dirname + '/' + fname)

def load_dt_policy(dirname, fname):
    f = open(dirname + '/' + fname, 'rb')
    dt_policy = pk.load(f)
    f.close()
    return dt_policy

class DTPolicy(Agent):
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        self.fit(obss_train, acts_train)
        info = {
            'train_acc': accuracy(self, obss_train, acts_train),
            'test_acc': accuracy(self, obss_test, acts_test),
            'num_nodes': self.tree.tree_.node_count
        }
        return info

    def predict(self, obss):
        return self.tree.predict(obss)

    def get_action(self, obs):
        action = self.predict([obs])[0]
        return action

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone