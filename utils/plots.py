import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

def line_plot(x, y, title, xlabel, ylabel, save_path=None):
    """Plot line graph."""
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    return fig
    
def create_grids(agent, usable_ace=False):
    """Create value and policy grid given an agent."""
    # convert our state-action values to state values
    # and build a policy dictionary that maps observations to actions
    state_value = defaultdict(float)
    policy = defaultdict(int)
    print("shape", agent.q_values.shape)
    for player_count in range(agent.q_values.shape[0]):
        for dealer_count in range(agent.q_values.shape[1]):
            for usable_ace in range(agent.q_values.shape[2]):
                obs = (player_count, dealer_count, usable_ace)
                action_values = agent.q_values[obs]
                state_value[obs] = float(np.max(action_values))
                policy[obs] = int(np.argmax(action_values))

    # for obs, action_values in agent.q_values:
    #     state_value[obs] = float(np.max(action_values))
    #     policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        # players count, dealers face-up card
        np.arange(12, 22),
        np.arange(1, 11),
    )

    # create the value grid for plotting
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # create the policy grid for plotting
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    return value_grid, policy_grid

def create_plots(value_grid, policy_grid, title: str):
    """Creates a plot using a value and policy grid."""
    # create a new figure with 2 subplots (left: state values, right: policy)
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16)

    # plot the state values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State values: {title}")
    ax1.set_xlabel("Player sum")
    ax1.set_ylabel("Dealer showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # plot the policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player sum")
    ax2.set_ylabel("Dealer showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    # add a legend
    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
    return fig

def plot_table_blackjack(data, center=None, figsize=(7.5, 12), cmap=None, title=""):
    '''
    Flatten from 4-D to 2-D and plot all heatmaps.
    '''
    
    TITLE = ['Stick, No Usable Ace', 'Stick, With Usable Ace', 'Hit, No Usable Ace', 'Hit, With Usable Ace']
    cmap = 'Blues' if cmap is None else cmap

    # f, ax = plt.subplots(figsize=figsize)
    nrows = 2
    ncols = 2
    f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows), constrained_layout=True)
    f.suptitle(title, fontsize=16)
    
    to_plot = np.split(data, data.shape[-1], axis=-1)
    to_plot = [np.squeeze(d) for d in to_plot]
    
    to_plot = [np.split(d, d.shape[-1], axis=-1) for d in to_plot]
    to_plot = [np.squeeze(t) for sub in to_plot for t in sub]
    for idx, (ax, plot) in enumerate(zip(axes.flatten(), to_plot)):
        sns.heatmap(plot, center=center, linewidth=1, yticklabels=1, cmap=cmap, ax=ax, cbar_kws={"fraction": 0.1, "pad": 0.1, "aspect": 40})
        ax.set_title(TITLE[idx])
        # States outside this range are unreachable
        ax.set_ylim(22, 4)
        ax.set_xlim(1, 11)
        ax.set_ylabel('Sum of Player Hand')
        ax.set_xlabel('Dealer Face-up Card')
        ax.tick_params(labelsize=10)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)
    return f

def plot_table_cartpole(data, title, figsize=(20, 4), contrast=False):
    center = None
    if contrast:
        cmap = sns.diverging_palette(10, 240, n=128)
        center = 0
    else:
        cmap = 'Blues'

    f, ax = plt.subplots(figsize=figsize)
    data = data.transpose()
    ax = sns.heatmap(data, linewidth=1, center=center, cmap=cmap, xticklabels=20, yticklabels=['left', 'right'])
    ax.set_title(title)
    ax.set_ylim(2, 0)
    ax.set_ylabel('Actions')
    ax.set_xlabel('States')
    # ax.set_xticks(np.arange(70, 91, 1))
    ax.tick_params(labelsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # f.tight_layout()
    f.subplots_adjust(top=0.8, bottom=0.3, right=0.8)
    f.tight_layout()
    return f