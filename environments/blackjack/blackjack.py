from environments.env_instance import EnvironmentInstance
from utils.plot import create_grids, create_plots, plot_table_blackjack
import seaborn as sns
import matplotlib.pyplot as plt


class BlackjackEnvironment(EnvironmentInstance):
    def __init__(self, **kwargs):
        super().__init__("Blackjack-v1", **kwargs)

    def generate_plots(self, agent, evaluation_results, deep=False):
        """Generates plots for the given agent and evaluation results in the Blackjack Environment.
            'deep' is a boolean flag to indicate whether the agent is a DQN agent or not."""
        super().generate_plots(agent, evaluation_results)

        # # state values & policy with usable ace (ace counts as 11)
        # value_grid, policy_grid = create_grids(agent, usable_ace=True)
        # fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
        # plt.show()

        # # state values & policy without usable ace (ace counts as 1)
        # value_grid, policy_grid = create_grids(agent, usable_ace=False)
        # fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
        # plt.show()

        table_cmap = sns.diverging_palette(10, 240, n=128)
        fig3 = plot_table_blackjack(agent.q_values, center = 0, cmap=table_cmap, title="Q-Values")
        # save fig3
        plt.savefig("images/blackjack_q_values.png")
        if not deep:
            # currently no H-values implementation for DQN
            fig4 = plot_table_blackjack(agent.h_values, center = 0, cmap=table_cmap, title="H-Values")
            # save fig4
            plt.savefig("images/blackjack_h_values.png")
        # plt.show()