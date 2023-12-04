from environments.env_instance import EnvironmentInstance
from utils.plots import create_grids, create_plots, plot_table_blackjack
import seaborn as sns
import matplotlib.pyplot as plt


class BlackjackEnvironment(EnvironmentInstance):
    def __init__(self, **kwargs):
        super().__init__("Blackjack-v1", **kwargs)

    def generate_plots(self, evaluation_results, agent, deep=False):
        """Generates plots for the given agent and evaluation results in the Blackjack Environment.
            'deep' is a boolean flag to indicate whether the agent is a DQN agent or not."""
        if agent is None:
            raise("Generate plots function for Blackjack environment requires agent to be passed.")
        super().generate_plots(evaluation_results)

        # # state values & policy with usable ace (ace counts as 11)
        # value_grid, policy_grid = create_grids(agent, usable_ace=True)
        # fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
        # plt.show()

        # # state values & policy without usable ace (ace counts as 1)
        # value_grid, policy_grid = create_grids(agent, usable_ace=False)
        # fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
        # plt.show()
        
        q_data=None
        h_data=None
        if not deep:
            q_data = agent.q_values
            h_data = agent.h_values
        else:
            q_data = agent.generate_q_table(self._instance)

        if q_data is not None:
            table_cmap = sns.diverging_palette(10, 240, n=128)
            plot_table_blackjack(q_data, center = 0, cmap=table_cmap, title="Q-Values")
            # save fig3
            plt.savefig("images/blackjack_q_values.png")
            
        if h_data is not None:
            # currently no H-values implementation for DQN
            plot_table_blackjack(h_data, center = 0, cmap=table_cmap, title="H-Values")
            # save fig4
            plt.savefig("images/blackjack_h_values.png")
        # plt.show()