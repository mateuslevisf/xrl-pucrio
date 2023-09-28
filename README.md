# xrl-pucrio

Creating environment from environment.yml file:
```
conda env create -f environment.yml
```

To activate resulting environment: 
```
conda activate xrlpucrio
```

The `play.py` inside the `GymTutorial` folder is currently not runnable because of a problem with installing box2d (in theory it's not compatible with python 3.9 but couldn't make it work for python 3.8 either).