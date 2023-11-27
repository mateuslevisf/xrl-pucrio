## Play Gym Blackjack 
# import gymnasium as gym
# from gym.utils import play

# env = gym.make("Blackjack-v1", render_mode="human", sab="True")

# play.play(env, fps=10, zoom=1, callback=None, keys_to_action=None)

## Code below is from "play" Gym util tutorial but doesn't work because of Box2d dependency
# import gymnasium as gym
# import numpy as np
# from gymnasium.utils.play import play
# play(gym.make("CarRacing-v2", render_mode="rgb_array"), keys_to_action={  
#                                                "w": np.array([0, 0.7, 0]),
#                                                "a": np.array([-1, 0, 0]),
#                                                "s": np.array([0, 0, 1]),
#                                                "d": np.array([1, 0, 0]),
#                                                "wa": np.array([-1, 0.7, 0]),
#                                                "dw": np.array([1, 0.7, 0]),
#                                                "ds": np.array([1, 0, 1]),
#                                                "as": np.array([-1, 0, 1]),
#                                               }, noop=np.array([0,0,0]))