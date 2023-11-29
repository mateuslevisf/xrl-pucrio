class BlackjackAgent:
    def __init__(self):
        raise(NotImplementedError)

    def get_action(self, obs: tuple[int, int, bool]) -> int:
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




