import numpy as np

class BlackjackAgent:
    def __init__(self):
        raise(NotImplementedError)

    def get_action(self, obs: tuple[int, int, bool], eval: bool = False) -> int:
        """Returns the action to be taken given the current state of the environment. Should always be an index of the action space."""
        raise(NotImplementedError)

    def update(
        self,
        obs: tuple[int, int, bool],
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

