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

        # Perform mean-shift algorithm to initialize the parameter of the posterior over latent variables.
        clustering = Clustering(bandwidth=bandwidth).fit(x)
        r = clustering.labels_
        n_classes = max(r) + 1
        if n_classes > n_states:
            print("[Warning] The number of classes found by MeanShift is superior to the number of states.")
        m = MeanShift.mean(x, r, n_classes)
        r = one_hot(torch.from_numpy(r), n_classes)

        # Initialize the degrees of freedom, as well as the scale and Dirichlet parameters.
        v = (n_observations - 0.99) * torch.ones([n_classes])
        d = torch.ones([n_classes])
        β = torch.ones([n_classes])

        # Estimate the covariance of the clusters and use it to initialize the Wishart prior and posterior.
        precision = KMeans.precision(x, r)
        W = [precision[k] / v[k] for k in range(n_classes)]
        return v, d, β, m, W, r

    @staticmethod
    def mean(x, r, n_states):

        # Create the mean vector to be returned.
        m = torch.zeros([n_states, x.shape[1]])

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
