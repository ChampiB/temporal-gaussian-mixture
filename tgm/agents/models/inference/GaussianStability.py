import math
from copy import deepcopy

import torch
from torch import matmul, trace, logdet

from tgm.agents.models.inference.GaussianMixture import GaussianMixture as GMix


class GaussianStability:

    def __init__(self, kl_threshold=0.5, n_steps_threshold=4):
        self.kl_threshold = kl_threshold
        self.n_steps_threshold = n_steps_threshold
        self.fixed_components = []
        self.fixed_n_steps = None
        self.m_hat = self.W_hat = self.v_hat = self.N = None

    def update(self, gm):

        # Update number of steps for which components have remained fixed, and the list of fixed components.
        self.fixed_n_steps, self.fixed_components = self.compute_fixed_n_steps(
            self.fixed_n_steps, self.N, self.m_hat, self.W_hat, self.v_hat, gm.N, gm.m_hat, gm.W_hat, gm.v_hat
        )

        # Keep track of the (old) posterior's parameters.
        self.v_hat, _, _, self.m_hat, self.W_hat = GMix.clone(gm.v_hat, gm.d_hat, gm.β_hat, gm.m_hat, gm.W_hat)
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
            return []

        new_indices = []
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
                new_indices.append(min_index)

        return new_indices

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
        gs.m_hat = None if self.m_hat is None else [W_k.clone() for W_k in self.W_hat]
        gs.v_hat = None if self.v_hat is None else self.v_hat.clone()
        return gs
