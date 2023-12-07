# xrl-pucrio

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

The -h option can be added to the above command line in order to get more info about running options.

To run tests:
```
python -m unittest discover -v
```

## References

This codebase was inspired by multiple sources and other repositories.

For general RL code (such as training loop), see the ["Solving Blackjack with QLearning" tutorial](https://gymnasium.farama.org/tutorials/training_agents/blackjack_tutorial/) by Gymnasium.

For the Belief Map/H-values technique, see the [What Did You Think Would Happen? Explaining Agent Behaviour Through Intended Outcomes](https://arxiv.org/abs/2011.05064) paper and the [rl-intetion](https://github.com/hmhyau/rl-intention/) repo for the paper source code.

For the VIPER technique, see the [Verifiable Reinforcement Learning via Policy Extraction
](https://arxiv.org/abs/1805.08328) paper and the [viper](https://github.com/obastani/viper) repo for the paper source code.