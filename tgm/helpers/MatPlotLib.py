import gc
import math
import matplotlib.pyplot as plt
import torch
from matplotlib.widgets import Button

from tgm.helpers.PlotsBuilder import PlotsBuilder


class MatPlotLib:
    """
    A helper class providing useful functions for interacting with matplotlib.
    """

    @staticmethod
    def close(fig=None):
        """
        Close the figure passed as parameter or the current figure
        :param fig: the figure to close
        """

        # Clear the current axes.
        plt.cla()

        # Clear the current figure.
        plt.clf()

        # Closes all the figure windows.
        plt.close('all')

        # Closes the matplotlib figure
        plt.close(plt.gcf() if fig is None else fig)

        # Forces the garbage collection
        gc.collect()

    @staticmethod
    def format_image(img):
        """
        Turn a 4d pytorch tensor into a 3d numpy array
        :param img: the 4d tensor
        :return: the 3d array
        """
        return torch.squeeze(img).detach().cpu().numpy()

    @staticmethod
    def save_figure(out_f_name, dpi=300, tight=True):
        """
        Save a matplotlib figure in an `out_f_name` file.
        :param str out_f_name: Name of the file used to save the figure.
        :param int dpi: Number of dpi, Default 300.
        :param bool tight: If True, use plt.tight_layout() before saving. Default True.
        """
        if tight is True:
            plt.tight_layout()
        plt.savefig(out_f_name, dpi=dpi, transparent=True)
        MatPlotLib.close()

    @staticmethod
    def draw_k_means(μ, data, r, title=""):
        """
        Draw the Gaussian Mixture graph
        :param μ: the cluster center
        :param data: the data points
        :param r: the responsibilities for all data points
        :param title: the title of the figure
        """

        # Draw the plots.
        plots = PlotsBuilder(title)
        plots.draw_k_means(title="Observation at t = 0", data=data, r=r, μ=μ)

        # Display the graph.
        plots.show()

    @staticmethod
    def draw_gm_graph(params, data, r, title="", clusters=False, ellipses=True, skip_fc=None):
        """
        Draw the Gaussian Mixture graph
        :param params: a 3-tuples of the form (m_hat, v_hat, W_hat)
        :param data: the data points
        :param r: the responsibilities for all data points
        :param title: the title of the figure
        :param clusters: whether to draw the cluster centers
        :param ellipses: whether to draw the ellipses
        :param skip_fc: the function to call when the skip button is clicked
        """

        # Draw the plots.
        plots = PlotsBuilder(title, n_cols=2)
        plots.draw_gaussian_mixture(
            title="Observation at t = 0", data=data, r=r, params=params, clusters=clusters, ellipses=ellipses
        )
        plots.draw_histograms(title="Responsibilities at t = 0", r=r)

        # Add the skip button.
        if skip_fc is not None:
            plt.gcf().add_axes([0.97, 0.97, 0.02, 0.02])
            b_next = Button(plt.gca(), 'Skip', color="limegreen", hovercolor="forestgreen")
            b_next.on_clicked(skip_fc)

        # Display the graph.
        plots.show()

    @staticmethod
    def draw_tgm_graph(action_names, params, x0, x1, a0, r, title="", clusters=False, ellipses=True):
        """
        Draw the Temporal Gaussian Mixture graph
        :param action_names: name of all the environment's actions
        :param params: a 3-tuples of the form (m_hat, v_hat, W_hat)
        :param a0: the actions at time step zero
        :param x0: the data points at time step zero
        :param x1: the data points at time step one
        :param r: the responsibilities for all data points at time steps zero and one
        :param title: the title of the figure
        :param clusters: whether to draw the cluster centers
        :param ellipses: whether to draw the ellipses
        """

        # Retrieve the number of actions.
        n_actions = len(action_names)

        # Create the plot builder.
        plots = PlotsBuilder(title, n_rows=1 + math.ceil(n_actions / 4.0), n_cols=4)

        # Draw the model's beliefs.
        plots.draw_gaussian_mixture(
            title="Observation at t = 0", data=x0, r=r[0], params=params, clusters=clusters, ellipses=ellipses
        )
        plots.draw_gaussian_mixture(
            title="Observation at t = 1", data=x1, r=r[1], params=params, clusters=clusters, ellipses=ellipses
        )

        # Draw the responsibilities.
        plots.draw_histograms(title="Responsibilities at t = 0", r=r[0])
        plots.draw_histograms(title="Responsibilities at t = 1", r=r[1])

        # Draw the transition graph for each action.
        for action in range(n_actions):
            plots.draw_transition_graph(action, a0, r, title=f"Transition for action = {action_names[action]}")

        # Show all the plots.
        plots.show()

    @staticmethod
    def draw_dirichlet_tmhgm_graph(
        action_names, params0, params1, x0, x1, a0, r, b_hat, title="", clusters=False, ellipses=True, skip_fc=None
    ):
        """
        Draw the Temporal Gaussian Mixture graph
        :param action_names: name of all the environment's actions
        :param params0: a 3-tuples of the form (m_hat, v_hat, W_hat)
        :param params1: a 3-tuples of the form (m_hat, v_hat, W_hat)
        :param a0: the actions at time step zero
        :param x0: the data points at time step zero
        :param x1: the data points at time step one
        :param r: the responsibilities for all data points at time steps zero and one
        :param b_hat: the non-normalized counts of the transition
        :param title: the title of the figure
        :param clusters: whether to draw the cluster centers
        :param ellipses: whether to draw the ellipses
        :param skip_fc: the function to call when the skip button is clicked
        """

        # Retrieve the number of actions.
        n_actions = len(action_names)

        # Create the plot builder.
        plots = PlotsBuilder(title, n_rows=1 + 2 * math.ceil(n_actions / 5.0), n_cols=5)

        # Draw the model's beliefs.
        plots.draw_gaussian_mixture(
            title="Observation at t = 0", data=x0, r=r[0], params=params0, clusters=clusters, ellipses=ellipses
        )
        plots.draw_gaussian_mixture(
            title="Observation at t = 1", data=x1, r=r[1], params=params1, clusters=clusters, ellipses=ellipses
        )

        # Draw the responsibilities.
        plots.draw_histograms(title="Responsibilities at t = 0", r=r[0])
        plots.draw_histograms(title="Responsibilities at t = 1", r=r[1])

        # Draw the transition graph for each action.
        for action in range(n_actions):
            plots.draw_transition_graph(action, a0, r, title=f"Transition for action = {action_names[action]}")

        # Draw the transition matrix for each action.
        for action in range(n_actions):
            plots.draw_matrix(b_hat[action], title=f"Transition matrix for action = {action_names[action]}")

        # Add the skip button.
        if skip_fc is not None:
            plt.gcf().add_axes([0.97, 0.97, 0.02, 0.02])
            b_next = Button(plt.gca(), 'Skip', color="limegreen", hovercolor="forestgreen")
            b_next.on_clicked(skip_fc)

        # Show all the plots.
        plots.show()

    @staticmethod
    def draw_action_distributions(action_names, a0, r0, title):

        # Create the plot builder.
        plots = PlotsBuilder(title, n_rows=math.ceil(r0.shape[1] / 4.0), n_cols=4)

        n_actions = len(action_names)
        a0 = torch.concat([torch.nn.functional.one_hot(a, num_classes=n_actions) for a in a0])

        states = r0.argmax(dim=1)
        for s in range(r0.shape[1]):
            keep = torch.tensor([state == int(s) for state in states])
            a = a0[keep]
            plots.draw_histograms(title=f"Distribution of actions taken in state {s}: ", r=a, x=action_names)

        # Show all the plots.
        plots.show()
