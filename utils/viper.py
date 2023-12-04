from environments.env_instance import EnvironmentInstance
from agents.agent import Agent

def get_rollout(env: EnvironmentInstance, agent: Agent) -> list: 
    """ 
    Runs a single rollout of the agent in the environment.
    """
    rollout = []
    obs, _ = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = next_obs

        rollout.append((obs, action, reward, done))
    
    return rollout

def get_rollout_batch(env: EnvironmentInstance, agent: Agent, rollout_batch_size: int) -> list:
    """Returns a batch of rollouts for use in VIPER technique."""
    batch = []

    for _ in range(rollout_batch_size):
        rollout = get_rollout(env, agent)
        batch.append(rollout)
    
    return batch

def train_viper(trained_agent: Agent, env: EnvironmentInstance, rollout_batch_size: int, num_policies: int):
    """Trains a decision tree on a trained agent (treated as an oracle) using the VIPER technique."""
    dataset = []

    for i in range(num_policies):
        # Get a batch of rollouts from the environment using the oracle
        rollout_batch = get_rollout_batch(env, trained_agent, rollout_batch_size)

        # Add the batch to the dataset
        dataset.append(rollout_batch)
    pass