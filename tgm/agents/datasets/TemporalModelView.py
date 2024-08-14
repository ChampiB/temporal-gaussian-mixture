from tgm.agents.datasets.Dataset import Dataset


class TemporalModelView:

    def __init__(self, dataset):
        self.dataset = dataset

    def get(self, gm, split=False):

        # Compute the responsibilities for each observation.
        r_hat = gm.compute_responsibilities(self.dataset.observations)

        # Return the entire dataset.
        if split is False:
            r0, r1, a0 = self._get_all(r_hat)
            r0 = Dataset.to_tensor(r0)
            r1 = Dataset.to_tensor(r1)
            a0 = Dataset.to_tensor(a0)
            return r0, r1, a0

        # Return the data points to forget and keep.
        (r0_forget, r1_forget, a0_forget), (r0_keep, r1_keep, a0_keep) = self._get_splits(r_hat)
        r0_forget = Dataset.to_tensor(r0_forget)
        r1_forget = Dataset.to_tensor(r1_forget)
        a0_forget = Dataset.to_tensor(a0_forget)
        r0_keep = Dataset.to_tensor(r0_keep)
        r1_keep = Dataset.to_tensor(r1_keep)
        a0_keep = Dataset.to_tensor(a0_keep)
        return (r0_forget, r1_forget, a0_forget), (r0_keep, r1_keep, a0_keep)

    def _get_all(self, r_hat):

        # Retrieve all the data points.
        r0 = []
        r1 = []
        a0 = []
        for i in range(len(self.dataset.x) - 1):
            if self.dataset.d[i] is False:
                a0.append(self.dataset.a[i])
                r0.append(r_hat[i])
                r1.append(r_hat[i + 1])
        return r0, r1, a0

    def _get_splits(self, r_hat):

        # Retrieve the data points to forget and keep.
        r0_forget = []
        r1_forget = []
        a0_forget = []
        r0_keep = []
        r1_keep = []
        a0_keep = []
        for i in range(len(self.dataset.x) - 1):
            if self.dataset.d[i] is False and self.dataset.ignore_next[i] is False:
                if self.dataset.forgettable[i] is True or self.dataset.forgettable[i + 1] is True:
                    a0_forget.append(self.dataset.a[i])
                    r0_forget.append(r_hat[i])
                    r1_forget.append(r_hat[i + 1])
                else:
                    a0_keep.append(self.dataset.a[i])
                    r0_keep.append(r_hat[i])
                    r1_keep.append(r_hat[i + 1])

        # Return the data points to forget and keep.
        return (r0_forget, r1_forget, a0_forget), (r0_keep, r1_keep, a0_keep)
