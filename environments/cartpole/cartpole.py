from environments.env_instance import EnvironmentInstance

class CartpoleEnvironment(EnvironmentInstance):
    def __init__(self, **kwargs):
        super().__init__("CartPole-v1", **kwargs)