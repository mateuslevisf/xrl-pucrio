
def save_params(env_params, agent_params):
    """Saves the parameters used in the experiment to a file."""
    env_params_file = open("env_params.txt", "w")
    env_params_file.write(str(env_params))
    env_params_file.close()

    agent_params_file = open("agent_params.txt", "w")
    agent_params_file.write(str(agent_params))
    agent_params_file.close()