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
        rollout = get_rollout(agent)
        batch.append(rollout)
    
    return batch
