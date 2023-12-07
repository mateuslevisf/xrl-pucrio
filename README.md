# xrl-pucrio

## Introduction

The goal of this project is to serve as both a basic introduction to the explainable reinforcement learning (XRL)
world and a sandbox for other XRL students/researchers to try out running XRL techniques with as little pre-running work
as possible. This was done through the implementation of two different XRL techniques (Belief Maps and VIPER) in a single codebase with non-technique-specific functions and classes being as generic as possible.

More information can be found in the repo wiki.

## Commands

Creating environment from environment.yml file:
```
conda env create -f conda_env.yml
```

To activate resulting environment: 
```
conda activate xrlpucrio
```

To run with default configuration:
```
python xrlpucrio.py
```

The -h option can be added to the above command line in order to get more info about running options. Note that results are saved in the "results" folder as the program is executed but the folder is completely erased at the start of each run - if the user wants to keep their results, they should be moved elsewhere on end of execution. 

To run all tests:
```
python -m unittest discover -v
```

The "main" tests (in files "test_run_hvalues.py" and "test_run_viper.py") take quite a while to run (around 10 minutes).

## References

This codebase was inspired by multiple sources and other repositories.

For general RL code (such as training loop), see the ["Solving Blackjack with QLearning" tutorial](https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/) by Gymnasium.

For the Belief Map/H-values technique, see the [What Did You Think Would Happen? Explaining Agent Behaviour Through Intended Outcomes](https://arxiv.org/abs/2011.05064) paper and the [rl-intention](https://github.com/hmhyau/rl-intention/) repo for the paper source code.

For the VIPER technique, see the [Verifiable Reinforcement Learning via Policy Extraction
](https://arxiv.org/abs/1805.08328) paper and the [viper](https://github.com/obastani/viper) repo for the paper source code.