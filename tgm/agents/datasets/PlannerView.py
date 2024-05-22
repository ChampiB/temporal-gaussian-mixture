class PlannerView:

    def __init__(self, dataset):
        self.dataset = dataset

    def get(self, gm):

        # Compute the responsibilities for each observation.
        r_hat = gm.compute_responsibilities(self.dataset.observations)

        # Retrieve all the data points.
        states = []
        actions = []
        rewards = []
        dones = []
        for i in range(len(self.dataset.x) - 1):
            if self.dataset.d[i] is False:
                actions.append(self.dataset.a[i])
                states.append(r_hat[i])
                rewards.append(self.dataset.r[i + 1])
                dones.append(self.dataset.d[i + 1])
        return states, actions, rewards, dones
