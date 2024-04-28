from torch import zeros

from tgm.agents.datasets.Dataset import Dataset


class GaussianMixtureView:

    def __init__(self, dataset):
        self.dataset = dataset

    def get(self, split=False):

        # Return the entire dataset.
        if split is False:
            return self.dataset.observations

        # Retrieve the data points to forget and keep.
        x_forget = []
        x_keep = []
        for x, forgettable in zip(self.dataset.x, self.dataset.forgettable):
            x_forget.append(x) if forgettable else x_keep.append(x)

        # Format the data points to forget and keep.
        n_observations = self.dataset.x[0].shape[0]
        x_forget = Dataset.to_tensor(x_forget) if len(x_forget) != 0 else zeros([0, n_observations])
        x_keep = Dataset.to_tensor(x_keep) if len(x_keep) != 0 else zeros([0, n_observations])

        # Return the data points to forget and keep.
        return x_forget, x_keep
