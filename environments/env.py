import gymnasium as gym
import tqdm

class Environment:
    def __init__(self, name, **kwargs):
        """Creates an environment instance; name should be Gym env name and 
        kwargs should be env arguments to be passed to gym.make() call."""
        self.name = name
        self.instance = gym.make(name, kwargs)
        self.observation_space = self.instance.observation_space
        self.action_space = self.instance.action_space

    def reset(self):
        """Resets the environment and returns the initial observation."""
        return self.instance.reset()

    def loop(self, agent, num_episodes, evaluation_interval, evaluation_duration):
        """Executes the training loop for the given agent and number of episodes.
            Returns a dictionary with the evaluation results."""
        evaluation_results = {}
        self.instance = gym.wrappers.RecordEpisodeStatistics(self.instance, deque_size=num_episodes)
        for episode in tqdm(range(num_episodes)):
            obs, _ = self.instance.reset()
            done = False

            while not done:
                action = agent.get_action(obs)
                # log("Action: {} in Episode: {}".format(action, episode))
                next_obs, reward, terminated, truncated, info = self.instance.step(action)

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
            obs, _ = self.instance.reset()
            done = False

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.instance.step(action)
                done = terminated or truncated
                obs = next_obs
                total_reward += reward

        return total_reward / num_episodes