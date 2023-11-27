class BlackjackAgent:
    def __init__(self):
        raise(NotImplementedError)

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        raise(NotImplementedError)

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        raise(NotImplementedError)




