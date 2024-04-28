import torch


class Dataset:

    def __init__(self):

        # Lists storing the observations, actions, rewards, and whether the episode ended.
        self.x = []
        self.a = []
        self.r = []
        self.d = []

        # List storing whether the points can be forgotten.
        self.forgettable = []

    def start_new_trial(self, obs):
        self.x.append(obs)
        self.a.append(None)
        self.r.append(None)
        self.d.append(False)
        self.forgettable.append(False)

    def append(self, obs, action, reward, done):
        self.x.append(obs)
        self.a[-1] = action
        self.a.append(None)
        self.r.append(reward)
        self.d.append(done)
        self.forgettable.append(False)

    @property
    def observations(self):
        return self.to_tensor(self.x)

    @staticmethod
    def to_tensor(xs):
        if len(xs) == 0:
            return torch.tensor([])
        if isinstance(xs[0], torch.Tensor):
            return torch.concat([x.unsqueeze(dim=0) for x in xs])
        if isinstance(xs[0], int) or isinstance(xs[0], float):
            return torch.concat([torch.tensor([x]).unsqueeze(dim=0) for x in xs], dim=0)
        raise TypeError("Unsupported type was sent to the Dataset.to_tensor function.")

    def update_forgettable_set(self, gm):

        # Compute the most likely components for each observation.

        r_hat = gm.compute_responsibilities(self.observations)
        z = r_hat.argmax(dim=1)

        # Retrieve the indices of all fixed components.
        fixed_ks = gm.fixed_components

        # Check whether each observation is associated to a fixed component.
        is_fixed = [True if z[i] in fixed_ks else False for i in range(z.shape[0])]

        # Check which entry can be discarded.
        forgettable = []
        for i in range(len(self.x)):
            previous_is_fixed = True if self.is_start_of_trial(i) or is_fixed[i - 1] else False
            future_is_fixed = True if self.is_end_of_trial(i) or (i != len(self.x) - 1 and is_fixed[i + 1]) else False
            forgettable.append(previous_is_fixed and is_fixed[i] and future_is_fixed)
        self.forgettable = forgettable

    def is_start_of_trial(self, i):
        return i == 0 or self.d[i - 1] is True

    def is_end_of_trial(self, i):
        return self.d[i] is True

    def forget(self):

        # Lists that will store the data points to keep.
        new_x = []
        new_a = []
        new_r = []
        new_d = []
        new_forgettable = []

        # Add to the data points to keep in the lists, i.e., forget the rest.
        update_done = False
        for x, a, r, d, forgettable in reversed(list(zip(self.x, self.a, self.r, self.d, self.forgettable))):
            if forgettable is False:
                new_x.insert(0, x)
                new_a.insert(0, a)
                new_r.insert(0, r)
                new_d.insert(0, True if update_done is True else d)
                new_forgettable.insert(0, forgettable)
                update_done = False
            else:
                update_done = True

        # Update the data points.
        self.x = new_x
        self.a = new_a
        self.r = new_r
        self.d = new_d
        self.forgettable = new_forgettable
