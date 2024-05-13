import matplotlib as mpl
import gc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import matplotlib.patheffects as pe
import torch
from matplotlib.patches import Rectangle


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
    def colors():

        # Create the list of colors to use.
        all_colors = list(colors.CSS4_COLORS.keys())
        first_colors = ['red', 'green', 'blue', 'purple', 'gray', 'pink', 'turquoise', 'orange', 'brown', 'cyan']
        for i, color in enumerate(first_colors):
            index = all_colors.index(color)
            i_color = all_colors[i]
            all_colors[i] = color
            all_colors[index] = i_color

        return all_colors

    @staticmethod
    def draw_k_means(μ, data, r, title=""):
        """
        Draw the Gaussian Mixture graph
        :param μ: the cluster center
        :param data: the data points
        :param r: the responsibilities for all data points
        :param title: the title of the figure
        """

        # Create a figure and the list of colors.
        fig = plt.figure()
        plt.gca().set_title(title)
        all_colors = MatPlotLib.colors()

        # Draw the data points.
        if data is not None:

            # Draw the data points of t = 0.
            x = [x_tensor[0] for x_tensor in data]
            y = [x_tensor[1] for x_tensor in data]

            r = torch.softmax(r, dim=1)
            c = [tuple(r_n) for r_n in r] if r.shape[1] == 3 else [all_colors[torch.argmax(r_n)] for r_n in r]
            plt.gca().scatter(x=x, y=y, c=c, s=3)

        # Draw the cluster center.
        x = [μ_k[0] for μ_k in μ]
        y = [μ_k[1] for μ_k in μ]
        plt.gca().scatter(x=x, y=y, marker="X", s=100, c="black", edgecolor="white")
        plt.gca().set_aspect('equal', adjustable='box')

        # Return the figure.
        return fig

    @staticmethod
    def draw_gaussian_mixture(
        data, params, r=None, title="", clusters=False, ellipses=True, active_only=True, display_ids=False, datum=None
    ):
        """
        Draw the Gaussian Mixture graph
        :param params: a string defining the parameter to use (i.e, prior, empirical_prior, posterior)
        :param data: the data points
        :param r: the responsibilities for all data points
        :param title: the title of the figure
        :param clusters: whether to draw the cluster centers
        :param ellipses: whether to draw the ellipses
        :param active_only: whether to display only the active components
        :param display_ids: whether to display the indices of the components on the ellipses
        :param datum: a data point that must be highlighted
        """

        # Create a figure and the list of colors.
        fig = plt.figure()
        plt.gca().set_title(title)
        all_colors = MatPlotLib.colors()

        # Draw the data points.
        if data is not None:

            # Draw the data points of t = 0.
            x = [x_tensor[0] for x_tensor in data]
            y = [x_tensor[1] for x_tensor in data]

            if r is None:
                c = [all_colors[0] for _ in x]
            else:
                r = torch.softmax(r, dim=1)
                c = [tuple(r_n) for r_n in r] if r.shape[1] == 3 else [all_colors[torch.argmax(r_n)] for r_n in r]
            plt.gca().scatter(x=x, y=y, c=c)

        # Highlight a data point.
        if datum is not None:
            x = [datum[0][0]]
            y = [datum[0][1]]
            plt.gca().scatter(x=x, y=y, marker="X", s=100, c="black", edgecolor="white")

        # Draw the ellipses corresponding to the current model believes.
        if ellipses is True:
            if r is None:
                active_components = {k for k in range(len(params[3]))}
            else:
                active_components = set(r.argmax(dim=1).tolist())
            MatPlotLib.make_ellipses(active_components, params, all_colors, active_only, display_ids)

        # Draw the cluster center.
        if clusters is True:
            μ, _, _, _ = params
            x = [μ_k[0] for μ_k in μ]
            y = [μ_k[1] for μ_k in μ]
            plt.gca().scatter(x=x, y=y, marker="X", s=100, c="black", edgecolor="white")

        plt.gca().set_aspect('equal', adjustable='box')

        # Return the figure.
        return fig

    @staticmethod
    def draw_fixed_components(
        data, fixed_components, params, r, counts=None, title="", clusters=False, ellipses=True, active_only=True
    ):
        """
        Draw the Gaussian Mixture graph
        :param data: the data points
        :param params: a string defining the parameter to use (i.e, prior, empirical_prior, posterior)
        :param fixed_components: the components which are fixed
        :param r: the responsibilities for all data points
        :param counts: the counts to display
        :param title: the title of the figure
        :param clusters: whether to draw the cluster centers
        :param ellipses: whether to draw the ellipses
        :param active_only: whether to display only the active components
        """

        # Create a figure and the list of colors.
        fig = plt.figure()
        plt.gca().set_title(title)
        fixed_color = "black"
        flexible_color = "red"

        # Draw the data points.
        if data is not None:

            # Draw the data points of t = 0.
            x = [x_tensor[0] for x_tensor in data]
            y = [x_tensor[1] for x_tensor in data]

            r_star = r.argmax(dim=1)
            c = [fixed_color if r_n in fixed_components else flexible_color for r_n in r_star]
            plt.gca().scatter(x=x, y=y, c=c)

        # Draw the ellipses corresponding to the current model believes.
        if ellipses is True:
            if r is None:
                active_components = {k for k in range(len(params[3]))}
            else:
                active_components = set(r.argmax(dim=1).tolist())
            all_colors = [fixed_color if k in fixed_components else flexible_color for k in range(len(params[3]))]
            MatPlotLib.make_ellipses(active_components, params, all_colors, active_only, counts=counts)

        # Draw the cluster center.
        if clusters is True:
            μ, _, _, _ = params
            x = [μ_k[0] for μ_k in μ]
            y = [μ_k[1] for μ_k in μ]
            plt.gca().scatter(x=x, y=y, marker="X", s=100, c="black", edgecolor="white")

        plt.gca().set_aspect("equal", adjustable="box")

        # Return the figure.
        return fig

    @staticmethod
    def draw_data_points(data, title="", data_colors="gray"):

        # Create a figure and the list of colors.
        fig = plt.figure()
        plt.gca().set_title(title)

        # Ensure the data and colors are lists.
        if not isinstance(data, list):
            data = [data]
        if not isinstance(data_colors, list):
            data_colors = [data_colors]

        # Draw the data points.
        for datum, color in zip(data, data_colors):
            x = [x_tensor[0] for x_tensor in datum]
            y = [x_tensor[1] for x_tensor in datum]
            plt.gca().scatter(x=x, y=y, c=color)

        # Return the figure.
        return fig

    @staticmethod
    def draw_ellipses(
        data, active_components, params, all_colors, title="", active_only=True, display_ids=False, counts=None
    ):

        # Create a figure and the list of colors.
        fig = plt.figure()
        plt.gca().set_title(title)

        # Draw the data points of t = 0.
        x = [x_tensor[0] for x_tensor in data]
        y = [x_tensor[1] for x_tensor in data]
        plt.gca().scatter(x=x, y=y, c="gray")

        # Display the ellipses in the figure.
        MatPlotLib.make_ellipses(active_components, params, all_colors, active_only, display_ids, counts)

        # Return the figure.
        return fig

    @staticmethod
    def make_ellipses(active_components, params, all_colors, active_only=True, display_ids=False, counts=None):

        m_hat, _, W_hat, v_hat = params
        for k in range(len(m_hat)):
            if active_only is True and k not in active_components:
                continue
            color = all_colors[k]

            covariances = torch.inverse(v_hat[k] * W_hat[k])
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi
            v = 3. * np.sqrt(2.) * np.sqrt(np.maximum(v, 0))
            mean = m_hat[k]
            mean = mean.reshape(2, 1)
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(plt.gca().bbox)
            ell.set_alpha(0.5)
            plt.gca().add_artist(ell)
            plt.gca().set_aspect("equal", "datalim")

            if display_ids is True:
                plt.text(
                    mean[0], mean[1], str(k), c=color, fontsize=20,
                    horizontalalignment="center", verticalalignment="center",
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")]
                )

            if counts is not None:
                plt.text(
                    mean[0], mean[1], str(counts[k]), c=color, fontsize=20,
                    horizontalalignment="center", verticalalignment="center",
                    path_effects=[pe.withStroke(linewidth=4, foreground="white")]
                )

    @staticmethod
    def draw_histograms(r, title=""):
        """
        Draw the Gaussian Mixture graph
        :param r: the responsibilities for all data points
        :param title: the title of the figure
        """

        # Create a figure and the list of colors.
        fig = plt.figure()
        plt.gca().set_title(title)
        all_colors = MatPlotLib.colors()

        # Retrieve the dataset size and the number of states.
        n_states = r.shape[1]

        # Draw a bar plot representing how many point are attributed to each component.
        x = [state for state in range(n_states)]
        y = r.sum(dim=0).tolist()
        bars = plt.gca().bar(x, y, align='center')
        for state in range(n_states):
            bars[state].set_color(all_colors[state])
            bars[state].set_alpha(0.53)

        # Return the figure.
        return fig

    @staticmethod
    def draw_graph(x, y, title):

        # Create a figure and the list of colors.
        fig = plt.figure(figsize=(20, 13))
        plt.gca().set_title(title)

        # Draw the plot.
        plt.plot(x, y)

        # Return the figure.
        return fig

    @staticmethod
    def draw_text(text):

        # Create a figure and the list of colors.
        fig = plt.figure()
        plt.gca().set_title(text)

        # Return the figure.
        return fig

    @staticmethod
    def draw_equation(equation, bg_color, text_color, pad_x=0, pad_y=0, font_size=10, dpi=200):

        # Create a figure and the list of colors.
        fig = plt.figure(figsize=(25, 4), dpi=dpi)
        plt.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{{amsmath,amssymb,amsfonts}}"
        })
        fig.patch.set_facecolor(bg_color)
        fig.set_facecolor(bg_color)
        r = fig.canvas.get_renderer()

        # Draw the equation.
        text = plt.gca().text(
            0.5, 0.5, rf"${equation}$", color=text_color, fontsize=font_size,
            horizontalalignment="center", verticalalignment="center",
            bbox=dict(boxstyle="square", fc=bg_color, ec=bg_color, pad=10)
        )
        plt.gca().set_facecolor(bg_color)
        plt.axis("off")

        # Crop useful part of the image.
        bb = text.get_window_extent(renderer=r)
        fig.set_size_inches((bb.width + pad_x) / dpi, (bb.height + pad_y) / dpi)

        # Return the figure.
        return fig

    @staticmethod
    def draw_matrix(matrix, title="", draw_values=True, log_scale=False, mask=None):

        # Create a figure.
        fig = plt.figure()
        plt.gca().set_title(title)

        # Draw the matrix passed as parameters.
        if log_scale is True:
            axis_img = plt.matshow(matrix + 10e-6, fignum=0, norm=mpl.colors.LogNorm())
        else:
            axis_img = plt.matshow(matrix + 10e-6, fignum=0)
        plt.colorbar(axis_img)

        # Draw the matrix values, if requested.
        if draw_values is True:
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[0]):
                    c = matrix[j, i]
                    if c >= 1000:
                        c = str(int(float(c) / 1000)) + "K"
                    elif c >= 100:
                        c = str(int(c))
                    else:
                        c = str(round(float(c), 2))
                    plt.gca().text(i, j, c, va="center", ha="center", c="white")
                    if mask is not None and mask[j, i] == 1:
                        plt.gca().add_patch(Rectangle(
                            (i - 0.5, j - 0.5), 1, 1, fc=(0.5, 0.5, 0.5, 0.5), ec="orange", lw=1
                        ))

        # Return the figure.
        return fig
