import math
from random import randint
import torch
from tgm.agents.models.display.MatPlotLib import MatPlotLib


class KMeans:
    """
    Implement the K-means algorithm.
    """

    @staticmethod
    def init_gm(x, n_observations, n_states, debug=False):

        # Initialize the degrees of freedom, as well as the scale and Dirichlet parameters.
        v = (n_observations - 0.99) * torch.ones([n_states])
        d = torch.ones([n_states])
        β = torch.ones([n_states])

        # Perform K-means to initialize the parameter of the posterior over latent variables at time step 1.
        m, r = KMeans.run(x, n_states, debug=debug)

        # Estimate the covariance of the clusters and use it to initialize the Wishart prior and posterior.
        precision = KMeans.precision(x, r)
        W = [precision[k] / v[k] for k in range(n_states)]
        return v, d, β, m, W, r

    @staticmethod
    def run(x, n_clusters, threshold=0.05, debug=False):
        μ = [x[randint(0, len(x) - 1)] for _ in range(n_clusters)]
        r = torch.zeros([len(x), n_clusters])
        diff = math.inf
        if debug is True:
            MatPlotLib.draw_k_means(μ, x, r, title="Initial")

        while diff > threshold:
            r = KMeans.update_responsibilities(x, μ)
            if debug is True:
                MatPlotLib.draw_k_means(μ, x, r, title="After updating responsibilities")

            μ, diff = KMeans.update_means(x, r, μ)
            if debug is True:
                MatPlotLib.draw_k_means(μ, x, r, title="After updating means")

        if debug is True:
            MatPlotLib.draw_k_means(μ, x, r, title="Final")
        return μ, r

    @staticmethod
    def update_responsibilities(x, μ):
        r = torch.zeros([len(x), len(μ)])
        for n in range(len(x)):
            distances = [KMeans.distance(x[n], μ[k]) for k in range(len(μ))]
            best_k = distances.index(min(distances))
            r[n][best_k] = 1
        return r

    @staticmethod
    def distance(x1, x2):
        if isinstance(x1, list):
            x1 = torch.tensor(x1)
        if isinstance(x2, list):
            x2 = torch.tensor(x2)
        return torch.pow(x1 - x2, 2).sum().sqrt()

    @staticmethod
    def update_means(x, r, μ_old):
        μ = []
        sum_r = r.sum(dim=0)
        diff = 0
        for k in range(r.shape[1]):
            μ_k = sum([r[n][k] * x[n] for n in range(len(x))]) / (sum_r[k] + 0.00001)
            μ.append(μ_k)
            diff += KMeans.distance(μ_k, μ_old[k])
        return μ, diff

    @staticmethod
    def precision(x, r):
        precision = []
        n_states = r.shape[1]
        for k in range(n_states):
            x_k = [torch.unsqueeze(x[n], dim=0) for n in range(len(x)) if r[n][k] == 1]
            x_k = torch.concat(x_k, dim=0).t() if len(x_k) != 0 else torch.tensor([[]])
            if x_k.shape[1] >= n_states:
                precision.append(torch.inverse(torch.cov(x_k)))
            else:
                epsilon = 0.00001
                var = torch.var(x_k, dim=1) + epsilon if x_k.shape[1] >= 2 else epsilon
                precision.append(torch.inverse(torch.eye(x[0].shape[0]) * var))
        return precision
