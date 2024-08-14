import torch
from torch.nn.functional import one_hot

from tgm.agents.models.display.MatPlotLib import MatPlotLib


class TemporalModel:

    def __init__(self, n_actions, n_states):
        self.n_actions = n_actions
        self.n_states = n_states

        self.b = torch.ones([self.n_actions, n_states, n_states])
        self.b_bar = torch.ones([self.n_actions, n_states, n_states])
        self.b_hat = torch.ones([self.n_actions, n_states, n_states])

        self.fixed_gaussian = None

    def fit(self, gm, x_forget, x_keep):

        # Initialize the parameter of the prior.
        n_states = len(gm.m)
        b = torch.ones([self.n_actions, n_states, n_states])
        if self.fixed_gaussian is not None:

            # Keep in mind the parameter of the fixed components.
            fixed_components_0 = self.fixed_gaussian.fixed_components
            v0, _, _, m0, W0 = self.fixed_gaussian.parameters_of(fixed_components_0)
            new_indices_map = self.fixed_gaussian.new_indices_of(v0, m0, W0, gm.v_hat, gm.m_hat, gm.W_hat)

            for a in range(b.shape[0]):
                for j in range(b.shape[1]):
                    for k in range(b.shape[2]):
                        if j not in new_indices_map.keys() or k not in new_indices_map.keys():
                            continue
                        j_new = new_indices_map[j]
                        k_new = new_indices_map[k]
                        b[a][j_new][k_new] = self.b[a][j][k]
        self.b = b

        # Compute the parameters of the empirical prior.
        (r0_forget, r1_forget, a0_forget) = x_forget
        if a0_forget.shape[0] == 0:
            self.b_bar = self.b.clone()
        else:
            a0_forget = one_hot(a0_forget, num_classes=self.n_actions).squeeze(dim=1).float()
            self.b_bar = self.b + torch.einsum("na, nj, nk -> ajk", a0_forget, r1_forget, r0_forget)

        # Compute the parameters of the posterior.
        (r0_keep, r1_keep, a0_keep) = x_keep
        if a0_keep.shape[0] == 0:
            self.b_hat = self.b_bar.clone()
        else:
            a0_keep = one_hot(a0_keep, num_classes=self.n_actions).squeeze(dim=1).float()
            self.b_hat = self.b_bar + torch.einsum("na, nj, nk -> ajk", a0_keep, r1_keep, r0_keep)

        # Keep track of the fixed components.
        self.fixed_gaussian = gm.fixed_gaussian.clone()

    def update_prior_parameters(self):

        # Update the prior parameters using the empirical prior parameters.
        self.b = None if self.b_bar is None else self.b_bar.clone()

    def clone(self):

        # Create a new temporal model.
        tm = TemporalModel(self.n_actions, self.n_states)

        # Clone the parameters of the temporal model.
        tm.b = None if self.b is None else self.b.clone()
        tm.b_bar = None if self.b_bar is None else self.b_bar.clone()
        tm.b_hat = None if self.b_hat is None else self.b_hat.clone()

        # Clone the Gaussian stability.
        tm.fixed_gaussian = None if self.fixed_gaussian is None else self.fixed_gaussian.clone()

        return tm

    def B(self, d="posterior"):

        if d == "prior":
            B = self.b.clone()
        elif d == "empirical_prior":
            B = self.b_bar.clone()
        elif d == "posterior":
            B = self.b_hat.clone()
        else:
            raise RuntimeError(f"Unsupported distribution type '{d}'.")

        # Normalize the columns of all matrices.
        n_states = B.shape[1]
        sum_B = B.sum(dim=1, keepdims=True).repeat(1, n_states, 1)
        return B / sum_B

    def draw_b_matrix(self, action, d="posterior", action_names=None, dirichlet_params=False):

        if action >= self.b.shape[0]:
            return None

        action_name = action if action_names is None else action_names[action]
        fig_size = [7.2, 6.2]

        if d == "prior":
            b = self.b if dirichlet_params is True else self.B()
            return MatPlotLib.draw_matrix(
                b[action], title=f"Prior concentration coefficient, action = {action_name}.",
                fig_size=fig_size
            )

        if d == "empirical_prior":
            b = self.b_bar if dirichlet_params is True else self.B()
            return MatPlotLib.draw_matrix(
                b[action], title=f"Empirical prior concentration coefficient, action = {action_name}.",
                fig_size=fig_size
            )

        if d == "posterior":
            b = self.b_hat if dirichlet_params is True else self.B()
            return MatPlotLib.draw_matrix(
                b[action], title=f"Posterior concentration coefficient, action = {action_name}.",
                fig_size=fig_size
            )

        raise RuntimeError(f"Unsupported distribution type '{d}'.")
