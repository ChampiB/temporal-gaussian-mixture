import torch
from sklearn.cluster import MeanShift as Clustering
from torch.nn.functional import one_hot

from tgm.agents.models.inference.KMeans import KMeans


class MeanShift:
    """
    Implement the means-shift algorithm.
    """

    @staticmethod
    def init_gm(x, n_observations, n_states, bandwidth=0.5):

        # Initialize the degrees of freedom, as well as the scale and Dirichlet parameters.
        v = (n_observations - 0.99) * torch.ones([n_states])
        d = torch.ones([n_states])
        β = torch.ones([n_states])

        # Perform mean-shift algorithm to initialize the parameter of the posterior over latent variables.
        clustering = Clustering(bandwidth=bandwidth).fit(x)
        r = clustering.labels_
        m = MeanShift.mean(x, r, n_states)
        r = one_hot(torch.from_numpy(r), n_states)

        # Estimate the covariance of the clusters and use it to initialize the Wishart prior and posterior.
        precision = KMeans.precision(x, r)
        W = [precision[k] / v[k] for k in range(n_states)]
        return v, d, β, m, W, r

    @staticmethod
    def mean(x, r, n_states):

        # Create the mean vector to be returned.
        m = torch.ones([n_states, x.shape[1]])

        # Iterate over the states.
        for i in range(n_states):

            # Compute the sum of xs associated with the i-th state.
            n_samples = 0
            for j in range(len(r)):
                if r[j] == i:
                    m[i] += x[j]
                    n_samples += 1

            # Divide by the number of samples.
            if n_samples != 0:
                m[i] /= n_samples

        return m
