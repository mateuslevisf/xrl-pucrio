import gymnasium as gym
from tqdm import tqdm

from utils.plots import line_plot

class EnvironmentInstance:
    def __init__(self, name, deep=False, technique="", **kwargs):
        """Creates an environment instance; name should be Gym env name and 
        kwargs should be env arguments to be passed to gym.make() call."""
        self.name = name
        self.technique = technique
        self._instance = gym.make(name, **kwargs)
        self._observation_space = self._instance.observation_space
        self._action_space = self._instance.action_space

    def reset(self):
        """Resets the environment and returns the initial observation."""
        return self._instance.reset()

    def step(self, action):
        """Executes the given action and returns the next observation, reward, 
        terminated, truncated and info objects."""
        return self._instance.step(action)

    def loop(self, agent, num_episodes, evaluation_interval, evaluation_duration = 50, already_wrapped=False):
        """Executes the training loop for the given agent and number of episodes.
            Returns a dictionary with the evaluation results."""
        evaluation_results = {}

        if not already_wrapped:
            self._instance = gym.wrappers.RecordEpisodeStatistics(self._instance, deque_size=num_episodes)
            
        for episode in tqdm(range(num_episodes)):
            obs, _ = self._instance.reset()
            done = False

            # print("obs", obs)

            while not done:
                action = agent.get_action(obs)
                # log("Action: {} in Episode: {}".format(action, episode))
                next_obs, reward, terminated, truncated, info = self._instance.step(action)

                # Update agent
                agent.update(obs, action, reward, terminated, next_obs)

                # Update environment status (done or not) and observation
                done = terminated or truncated
                obs = next_obs

            agent.decay_epsilon()

            # Evaluate agent
            if episode % evaluation_interval == 0:
                # log("Evaluating agent in episode {}".format(episode))
                evaluation_results[episode] = self.evaluate(agent, evaluation_duration)

        return evaluation_results

    def evaluate(self, agent, num_episodes):
        """Evaluates the agent for the given number of episodes and returns the average reward."""
        total_reward = 0
        for _ in range(num_episodes):
            obs, _ = self._instance.reset()
            done = False

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self._instance.step(action)
                done = terminated or truncated
                obs = next_obs
                total_reward += reward

        return total_reward / num_episodes

    def generate_plots(self, evaluation_results, **kwargs):
        """Generates generic plots for the given agent and evaluation results."""
        line_plot(evaluation_results.keys(), evaluation_results.values(), title="Evaluation results", xlabel="Episode number", 
            ylabel="Average reward", save_path="results/images/{}_evaluation_results.png".format(self.name))

    def close(self):
        """Closes the environment."""
        self._instance.close()

    def get_action_space(self):
        """Returns the action space of the environment."""
        return self._action_space

    def get_observation_space(self):
        """Returns the observation space of the environment."""
        return self._observation_space