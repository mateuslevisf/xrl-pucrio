import numpy as np

class Agent:
    """Base class for a Reinforcement Learning agent."""
    def __init__(self):
        """Initializes the agent. Must be implemented by all subclasses."""
        raise(NotImplementedError)

    def get_action(self, obs, eval: bool = False) -> int:
        """Returns the action to be taken given the current state of the environment. Should always be an index of the action space."""
        raise(NotImplementedError)
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the agent with the state information."""
        raise(NotImplementedError)

    def evaluate(
        self,
        env,
        eval_episodes: int
    ):
        """Evaluates the agent on the environment for a given number of episodes."""
        rewards = []
        for _ in range(eval_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.get_action(obs, eval=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)
        return np.mean(rewards)

    def save(self, path: str):
        """Saves the agent's parameters to a file."""
        raise(NotImplementedError)

    def load(self, path: str):
        """Loads the agent's parameters from a file."""
        raise(NotImplementedError)

