import numpy as np

from environments.env_instance import EnvironmentInstance
from agents.agent import Agent
from agents.q_agent import QAgent
from utils.viper.decision_tree import DTPolicy

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
        batch.extend(rollout)
    
    return batch

def _sample(obss, acts, qs, max_samples, is_reweight=False):
    # Step 1: Compute probabilities
    ps = np.max(qs, axis=1) - np.min(qs, axis=1)
    ps = ps / np.sum(ps)

    # Step 2: Sample points
    if is_reweight:
        # According to p(s)
        idx = np.random.choice(len(obss), size=min(max_samples, np.sum(ps > 0)), p=ps)
    else:
        # Uniformly (without replacement)
        idx = np.random.choice(len(obss), size=min(max_samples, np.sum(ps > 0)), replace=False)    

    # Step 3: Obtain sampled indices
    return obss[idx], acts[idx], qs[idx]


def test_student(env, student, n_test_rollouts):
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        student_trace = get_rollout(env, student, False)
        cum_rew += sum((rew for _, _, rew in student_trace))
    return cum_rew / n_test_rollouts

def get_best_student(env, students_and_rewards, n_tests = 100):
    while len(students_and_rewards) > 1:
        sorted_students = sorted(students_and_rewards, key=lambda x: x[1], reverse=True)

        # ignore latter half of students
        n_students = int((len(students_and_rewards) + 1)/2)

        new_students = []

        for i in range(len(n_students)):
            policy, reward = sorted_students[i]
            new_rewards = test_student(env, policy, n_tests)
            new_students.append((policy, np.mean(new_rewards)))

        students_and_rewards = new_students

    return students_and_rewards[0][0]


def train_viper(trained_agent: QAgent, env: EnvironmentInstance, rollout_batch_size: int, max_iters: int):
    """Trains a decision tree on a trained agent (treated as an oracle) using the VIPER technique."""
    decision_tree = DTPolicy(max_depth=5)

    students = []

    trace = get_rollout_batch(env, trained_agent, rollout_batch_size)
    observations = [obs for obs, _, _ in trace]
    actions = [act for _, act, _ in trace]
    q_values = [q for _, _, q in trace]

    for _ in range(max_iters):
        current_obs, current_actions, current_q_values = _sample(observations, actions, q_values, max_samples=100)
        decision_tree.train(current_obs, current_actions, current_q_values)

        student_trace = get_rollout_batch(env, decision_tree, rollout_batch_size)
        student_observations = [obs for obs, _, _ in student_trace]

        oracle_qs = [trained_agent.get_q_values_for_obs(obs) for obs in student_observations]
        oracle_actions = [np.argmax(qs) for qs in oracle_qs]

        observations.extend(student_observations)
        actions.extend(oracle_actions)
        q_values.extend(oracle_qs)

        current_reward = sum((rew for _, _, rew in student_trace)) / rollout_batch_size

        students.append((decision_tree.clone(), current_reward))
        
    best_student = get_best_student(env, students)

    return best_student