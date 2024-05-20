import math
from copy import deepcopy

import torch
from torch import matmul, trace, logdet

from tgm.agents.models.display.MatPlotLib import MatPlotLib
from tgm.agents.models.inference.GaussianMixture import GaussianMixture as GMix


class GaussianStability:

    def __init__(self, kl_threshold=0.5, n_steps_threshold=4):
        self.kl_threshold = kl_threshold
        self.n_steps_threshold = n_steps_threshold
        self.fixed_components = []
        self.fixed_n_steps = None
        self.β_hat = self.d_hat = self.m_hat = self.W_hat = self.v_hat = self.N = None

    def reindex(self):
        """
        Reindex the components such that the fixed components have indices equal to 0, 1, ..., |fixed_components| - 1
        """

        # Retrieve the flexible components.
        flexible_components = [k for k in range(len(self.m_hat)) if k not in self.fixed_components]

        # Create the re-indexed parameters.
        n_states = len(self.m_hat)
        reindex_v = torch.zeros([n_states])
        reindex_d = torch.zeros([n_states])
        reindex_β = torch.zeros([n_states])
        reindex_m = []
        reindex_W = []
        reindex_N = torch.zeros([n_states])
        fixed_components = []
        fixed_n_steps = torch.ones_like(self.fixed_n_steps)

        # Extract the parameters of the fixed components.
        for j, component in enumerate(self.fixed_components):
            reindex_v[j] = self.v_hat[component]
            reindex_d[j] = self.d_hat[component]
            reindex_β[j] = self.β_hat[component]
            reindex_m.append(self.m_hat[component].clone())
            reindex_W.append(self.W_hat[component].clone())
            reindex_N[j] = self.N[component]
            fixed_components.append(j)
            fixed_n_steps[j] = self.fixed_n_steps[component]

        # Extract the parameters of the flexible components.
        shift = len(self.fixed_components)
        for j, component in enumerate(flexible_components):
            reindex_v[shift + j] = self.v_hat[component]
            reindex_d[shift + j] = self.d_hat[component]
            reindex_β[shift + j] = self.β_hat[component]
            reindex_m.append(self.m_hat[component].clone())
            reindex_W.append(self.W_hat[component].clone())
            reindex_N[shift + j] = self.N[component]
            fixed_n_steps[shift + j] = self.fixed_n_steps[component]

        # Update the internal parameters.
        self.v_hat = reindex_v
        self.d_hat = reindex_d
        self.β_hat = reindex_β
        self.m_hat = reindex_m
        self.W_hat = reindex_W
        self.N = reindex_N
        self.fixed_components = fixed_components
        self.fixed_n_steps = fixed_n_steps

    def update(self, gm):

        # Update number of steps for which components have remained fixed, and the list of fixed components.
        self.fixed_n_steps, self.fixed_components = self.compute_fixed_n_steps(
            self.fixed_n_steps, self.N, self.m_hat, self.W_hat, self.v_hat, gm.N, gm.m_hat, gm.W_hat, gm.v_hat
        )

        # Keep track of the (old) posterior's parameters.
        self.v_hat, self.d_hat, self.β_hat, self.m_hat, self.W_hat = \
            GMix.clone(gm.v_hat, gm.d_hat, gm.β_hat, gm.m_hat, gm.W_hat)
        self.N = gm.N.clone()

    def compute_fixed_n_steps(self, fixed_n_steps_0, N0, m0, W0, v0, N1, m1, W1, v1):

        # Initialize the new list of fixed component, and the number of steps for each fixed Gaussian component.
        fixed_n_steps_1 = torch.ones_like(v1)
        if fixed_n_steps_0 is None:
            return fixed_n_steps_1, self.fixed_components
        fixed_components_1 = []

        # Iterate over all pairs of new and old states.
        js_assigned = []
        for k in range(v0.shape[0]):
            if N0[k] == 0:
                continue
            precision0 = W0[k] * v0[k]
            for j in range(v1.shape[0]):
                if j in js_assigned or N1[j] == 0:
                    continue
                precision1 = W1[j] * v1[j]

                # If the KL-divergence between the Gaussian components is small enough.
                kl = self.kl_gaussian(m0[k], precision0, m1[j], precision1)
                if kl < self.kl_threshold:

                    # Increase the number of steps associated to this component.
                    fixed_n_steps_1[j] += fixed_n_steps_0[k]
                    js_assigned.append(j)

                    # If this component was fixed or the number of steps is greater than the threshold,
                    # add the component to the fixed components.
                    if k in self.fixed_components or fixed_n_steps_1[j] > self.n_steps_threshold:
                        if j not in fixed_components_1:
                            fixed_components_1.append(j)
                    break

        return fixed_n_steps_1, fixed_components_1

    def new_indices_of(self, v0, m0, W0, v1, m1, W1):
        if v0 is None or m0 is None or W0 is None:
            return {}

        new_indices = {}
        for k in range(v0.shape[0]):
            precision0 = W0[k] * v0[k]

            min_kl = math.inf
            min_index = -1
            for j in range(v1.shape[0]):
                precision1 = W1[j] * v1[j]

                # Save the smallest KL-divergence.
                kl = self.kl_gaussian(m0[k], precision0, m1[j], precision1)
                if kl < min_kl:
                    min_kl = kl
                    min_index = j

            # Add the index corresponding to the smallest KL-divergence.
            if min_index != -1 and min_kl < self.kl_threshold:
                new_indices[k] = min_index

        return new_indices

    def parameters_of(self, components):

        if len(components) == 0:
            return None, None, None, None, None

        # Extract the parameters of the fixed components.
        n_states = len(components)
        extracted_v = torch.zeros([n_states])
        extracted_d = torch.zeros([n_states])
        extracted_β = torch.zeros([n_states])
        extracted_m = []
        extracted_W = []
        for j, component in enumerate(components):
            extracted_v[j] = self.v_hat[component]
            extracted_d[j] = self.d_hat[component]
            extracted_β[j] = self.β_hat[component]
            extracted_m.append(self.m_hat[component].clone())
            extracted_W.append(self.W_hat[component].clone())

        return extracted_v, extracted_d, extracted_β, extracted_m, extracted_W

    @staticmethod
    def kl_gaussian(m0, precision0, m1, precision1):

        # Retrieve the number of states, i.e., dimensions.
        n_states = m0.shape[0]

        # Compute the covariance matrices and there log determinants.
        sigma0 = precision0.inverse()
        log_det_sigma0 = logdet(sigma0)
        log_det_sigma1 = logdet(precision1.inverse())

        # Compute the trace term.
        sigma_trace = trace(matmul(precision1, sigma0))

        # Compute the quadratic form.
        diff = m1 - m0
        quadratic = matmul(diff.t(), matmul(precision1, diff))
        return 0.5 * (log_det_sigma1 - log_det_sigma0 - n_states + sigma_trace + quadratic)

    def get(self):
        return self.fixed_components

    def clone(self):
        gs = GaussianStability(self.kl_threshold, self.n_steps_threshold)
        gs.fixed_components = deepcopy(self.fixed_components)
        gs.fixed_n_steps = None if self.fixed_n_steps is None else self.fixed_n_steps.clone()
        gs.m_hat = None if self.m_hat is None else [m_k.clone() for m_k in self.m_hat]
        gs.W_hat = None if self.W_hat is None else [W_k.clone() for W_k in self.W_hat]
        gs.v_hat = None if self.v_hat is None else self.v_hat.clone()
        gs.d_hat = None if self.d_hat is None else self.d_hat.clone()
        gs.β_hat = None if self.β_hat is None else self.β_hat.clone()
        return gs

    def compute_responsibilities(self, x):
        if self.d_hat is None or self.β_hat is None or self.m_hat is None or self.v_hat is None or self.W_hat is None:
            return None
        log_D = GMix.expected_log_D(self.d_hat)
        log_det_Λ = GMix.expected_log_det_Λ(self.W_hat, self.v_hat)
        return GMix.responsibilities(x, self.m_hat, self.β_hat, self.W_hat, self.v_hat, log_D, log_det_Λ)

    def draw_distribution(self, x, r, display_ids=True, ellipses=True, datum=None):

        if r is None:
            return MatPlotLib.draw_data_points(x, title="Posterior distribution.")

        params = (self.m_hat, self.β_hat, self.W_hat, self.v_hat)
        return MatPlotLib.draw_gaussian_mixture(
            x, params, r,
            title="Posterior distribution.", display_ids=display_ids, ellipses=ellipses, datum=datum
        )

    def draw_fixed_components(self, x, r):

        if r is None:
            return MatPlotLib.draw_data_points(x, title="Posterior distribution.")

        counts = self.fixed_n_steps
        if counts is not None:
            counts = [int(count.item()) for count in counts]

        params = (self.m_hat, self.β_hat, self.W_hat, self.v_hat)
        return MatPlotLib.draw_fixed_components(
            x, self.fixed_components, params, r, title="Posterior distribution.", counts=counts
        )
