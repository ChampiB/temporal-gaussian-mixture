import torch
from torch import matmul

from tgm.agents.datasets.Dataset import Dataset
from tgm.agents.models.display.MatPlotLib import MatPlotLib
from tgm.agents.models.inference.GaussianStability import GaussianStability
from tgm.agents.models.inference.KMeans import KMeans
from tgm.agents.models.inference.GaussianMixture import GaussianMixture as GMix
from tgm.agents.models.inference.MeanShift import MeanShift


class GaussianMixtureModel:

    def __init__(self, n_states, n_observations):

        # Keep track of the number of states and observations
        self.n_states = n_states
        self.n_observations = n_observations

        # The prior parameters.
        self.W = self.m = self.v = self.β = self.d = None

        # The empirical prior parameters.
        self.W_bar = self.m_bar = self.v_bar = self.β_bar = self.d_bar = self.r_bar = None

        # The posterior parameters.
        self.W_hat = self.m_hat = self.v_hat = self.β_hat = self.d_hat = self.r_hat = None

        # List of indices corresponding to fixed components.
        self.fixed_gaussian = GaussianStability(kl_threshold=0.5, n_steps_threshold=4)

        # Pre-computed terms.
        self.log_D = self.log_det_Λ = self.N = self.x_bar = self.S = self.vfe = None
        self.N_prime = self.x_prime = self.S_prime = None
        self.N_second = self.x_second = self.S_second = None

    def fit(self, x_forget, x_keep, split=True, threshold=1, debugger=None):

        n_steps = 0
        i = 0
        while n_steps < 5 and i < 100:

            # Keep track of previous variational free energy.
            vfe = self.vfe
            if vfe.isnan().any():
                print("[Warning] The VFE is 'not a number'.")
                break

            # Update for Z (forgettable data points).
            self.r_bar = self.compute_responsibilities(x_forget)
            self.N_prime = GMix.N(self.r_bar)
            self.x_prime = GMix.x_bar(x_forget, self.r_bar, self.N_prime)
            self.S_prime = GMix.S(x_forget, self.r_bar, self.N_prime, self.x_prime)

            # Update for Z (data points to keep).
            self.r_hat = self.compute_responsibilities(x_keep)
            self.N_second = GMix.N(self.r_hat)
            self.x_second = GMix.x_bar(x_keep, self.r_hat, self.N_second)
            self.S_second = GMix.S(x_keep, self.r_hat, self.N_second, self.x_second)

            # Update pre-computed terms.
            self.N = self.N_prime + self.N_second
            self.S = [
                self.weighted_average(
                    self.N_prime[k], self.S_prime[k], self.N_second[k], self.S_second[k], self.N[k]
                ) for k in range(len(self.S_prime))
            ]
            self.x_bar = [
                self.weighted_average(
                    self.N_prime[k], self.x_prime[k], self.N_second[k], self.x_second[k], self.N[k]
                ) for k in range(len(self.S))
            ]
            self.x_bar = Dataset.to_tensor(self.x_bar)
            if debugger is not None:
                debugger.after_update()

            # Update for D.
            self.d_bar = self.d + self.N_prime
            self.d_hat = self.d_bar + self.N_second
            self.log_D = GMix.expected_log_D(self.d_hat)
            if debugger is not None:
                debugger.after_update()

            # Update for μ and Λ.
            self.v_bar = self.v + self.N_prime
            self.v_hat = self.v_bar + self.N_second
            self.β_bar = self.β + self.N_prime
            self.β_hat = self.β_bar + self.N_second
            self.m_bar = GMix.m_hat(self.β, self.m, self.N_prime, self.x_prime, self.β_bar)
            self.m_hat = GMix.m_hat(self.β_bar, self.m_bar, self.N_second, self.x_second, self.β_hat)
            self.W_bar = GMix.W_hat(self.W, self.N_prime, self.S_prime, self.x_prime, self.m, self.β, self.β_bar)
            self.W_hat = GMix.W_hat(self.W_bar, self.N_second, self.S_second, self.x_second, self.m_bar, self.β_bar, self.β_hat)
            self.log_det_Λ = GMix.expected_log_det_Λ(self.W_hat, self.v_hat)
            if debugger is not None:
                debugger.after_update()

            # Update variational free energy and the number of steps.
            self.vfe = GMix.vfe(self)
            n_steps = n_steps + 1 if float(vfe - self.vfe) < threshold else 0

            # Keep track of the number of inference steps.
            i += 1

        # Try splitting the Gaussian components that are not fixed yet.
        if split is True:
            self.split_flexible_components(x_keep)

    @staticmethod
    def weighted_average(N_prime, x_prime, N_second, x_second, N):
        avg = torch.zeros_like(x_prime)
        if N_prime != 0:
            avg += N_prime * x_prime
        if N_second != 0:
            avg += N_second * x_second
        return avg / N

    def split_flexible_components(self, x_keep):
        pass  # TODO

    def is_initialized(self):
        return self.W is not None

    def initialize(self, x, init_type="mean-shift"):

        # Retrieve the parameters of the fixed components.
        v = d = β = m = W = None
        x_flexible = x
        if len(self.fixed_components) != 0:
            v, d, β, m, W = self.parameters_of(self.fixed_components)
            r_hat = self.compute_responsibilities(x)
            x_flexible = self.not_data_of(x, r_hat.argmax(dim=1), self.fixed_components)

        # Compute the prior parameters and the responsibilities using the k-means or mean-shift algorithms.
        if x_flexible is None:
            self.v, self.d, self.β, self.m, self.W = v, d, β, m, W
        else:
            init_fc = {
                "k-means": KMeans.init_gm,
                "mean-shift": MeanShift.init_gm
            }
            self.v, self.d, self.β, self.m, self.W, self.r_hat = \
                init_fc[init_type](x_flexible, self.n_observations, self.n_states)
            if v is not None:
                self.v, self.d, self.β, self.m, self.W = \
                    self.merge(v, d, β, m, W, self.v, self.d, self.β, self.m, self.W)

        # Initialize the empirical prior parameters.
        self.v_bar, self.d_bar, self.β_bar, self.m_bar, self.W_bar = GMix.clone(self.v, self.d, self.β, self.m, self.W)

        # Initialize the posterior parameters.
        self.v_hat, self.d_hat, self.β_hat, self.m_hat, self.W_hat = GMix.clone(self.v, self.d, self.β, self.m, self.W)

        # Initialize pre-computed terms.
        self.log_D = GMix.expected_log_D(self.d_hat)
        self.log_det_Λ = GMix.expected_log_det_Λ(self.W_hat, self.v_hat)
        self.r_hat = self.compute_responsibilities(x)
        self.N = GMix.N(self.r_hat)
        self.x_bar = GMix.x_bar(x, self.r_hat, self.N)
        self.S = GMix.S(x, self.r_hat, self.N, self.x_bar)
        self.vfe = GMix.vfe(self)

    def not_data_of(self, x, r_star, components, threshold=10):
        data = []
        for i in range(x.shape[0]):
            if r_star[i] in components:
                datum = x[i].unsqueeze(dim=0)
                if self.mahalanobis_distance(x[i], r_star[i]) > threshold:
                    data.append(datum)
        if len(data) == 0:
            return None
        return torch.concat(data, dim=0)

    def mahalanobis_distance(self, x, i):
        diff = (x - self.m_hat[i])
        return matmul(diff.t(), matmul(self.W_hat[i] * self.v_hat[i], diff))

    @staticmethod
    def merge(v0, d0, β0, m0, W0, v1, d1, β1, m1, W1):
        v0 = torch.concat([v0, v1], dim=0)
        d0 = torch.concat([d0, d1], dim=0)
        β0 = torch.concat([β0, β1], dim=0)
        m0.extend(m1)
        W0.extend(W1)
        return v0, d0, β0, m0, W0

    def parameters_of(self, fixed_components, d="prior"):

        # Retrieve all the parameters of interest.
        if d == "prior":
            v, d, β, m, W = self.v, self.d, self.β, self.m, self.W
        elif d == "empirical_prior":
            v, d, β, m, W = self.v_bar, self.d_bar, self.β_bar, self.m_bar, self.W_bar
        elif d == "posterior":
            v, d, β, m, W = self.v_hat, self.d_hat, self.β_hat, self.m_hat, self.W_hat
        else:
            raise RuntimeError(f"Unsupported distribution type '{d}'.")

        # Extract the parameters of the fixed components.
        n_states = len(fixed_components)
        v_fixed = torch.zeros([n_states])
        d_fixed = torch.zeros([n_states])
        β_fixed = torch.zeros([n_states])
        m_fixed = []
        W_fixed = []
        for j, fixed_component in enumerate(fixed_components):
            v_fixed[j] = v[fixed_component]
            d_fixed[j] = d[fixed_component]
            β_fixed[j] = β[fixed_component]
            m_fixed.append(m[fixed_component].clone())
            W_fixed.append(W[fixed_component].clone())

        return v_fixed, d_fixed, β_fixed, m_fixed, W_fixed

    def update_fixed_components(self):
        self.fixed_gaussian.update(self)

    def compute_responsibilities(self, x, d="posterior"):

        if d == "posterior":
            return GMix.responsibilities(x, self.m_hat, self.β_hat, self.W_hat, self.v_hat, self.log_D, self.log_det_Λ)

        if d == "empirical_prior":
            log_D = GMix.expected_log_D(self.d_bar)
            log_det_Λ = GMix.expected_log_det_Λ(self.W_bar, self.v_bar)
            return GMix.responsibilities(x, self.m_bar, self.β_bar, self.W_bar, self.v_bar, log_D, log_det_Λ)

        if d == "prior":
            log_D = GMix.expected_log_D(self.d)
            log_det_Λ = GMix.expected_log_det_Λ(self.W, self.v)
            return GMix.responsibilities(x, self.m, self.β, self.W, self.v, log_D, log_det_Λ)

        raise RuntimeError(f"Unsupported distribution type '{d}'.")

    @property
    def fixed_components(self):
        return self.fixed_gaussian.get()

    @property
    def flexible_components(self):
        fixed_components = self.fixed_gaussian.get()
        return [k for k in range(len(self.W)) if k not in fixed_components]

    @property
    def active_components(self):
        return set(self.r_hat.argmax(dim=1).tolist())

    def update_prior_parameters(self):

        # Update the prior parameters using the empirical prior parameters.
        self.v, self.d, self.β, self.m, self.W = GMix.clone(self.v_bar, self.d_bar, self.β_bar, self.m_bar, self.W_bar)

    def clone(self):
        
        # Create a new Gaussian mixture.
        gm = GaussianMixtureModel(self.n_states, self.n_observations)

        # Clone the prior parameters.
        gm.v, gm.d, gm.β, gm.m, gm.W = GMix.clone(self.v, self.d, self.β, self.m, self.W)

        # Clone the empirical prior parameters.
        gm.v_bar, gm.d_bar, gm.β_bar, gm.m_bar, gm.W_bar = \
            GMix.clone(self.v_bar, self.d_bar, self.β_bar, self.m_bar, self.W_bar)
        gm.r_bar = None if self.r_bar is None else self.r_bar.clone()

        # Clone the posterior parameters.
        gm.v_hat, gm.d_hat, gm.β_hat, gm.m_hat, gm.W_hat = \
            GMix.clone(self.v_hat, self.d_hat, self.β_hat, self.m_hat, self.W_hat)
        gm.r_hat = None if self.r_hat is None else self.r_hat.clone()

        # Clone the list of indices corresponding to fixed components.
        gm.fixed_gaussian = self.fixed_gaussian.clone()

        # Clone the pre-computed terms.
        gm.log_D = self.log_D.clone()
        gm.log_det_Λ = self.log_det_Λ.clone()
        gm.N = self.N.clone()
        gm.x_bar = self.x_bar.clone()
        gm.S = [S_k.clone() for S_k in self.S]
        gm.N_prime = None if self.N_prime is None else self.N_prime.clone()
        gm.x_prime = None if self.x_prime is None else self.x_prime.clone()
        gm.S_prime = None if self.S_prime is None else [S_k.clone() for S_k in self.S_prime]
        gm.N_second = None if self.N_second is None else self.N_second.clone()
        gm.x_second = None if self.x_second is None else self.x_second.clone()
        gm.S_second = None if self.S_second is None else [S_k.clone() for S_k in self.S_second]
        gm.vfe = GMix.vfe(self)

        return gm

    def draw_distribution(self, x, r, d="posterior", display_ids=True):

        if d == "prior":
            params = (self.m, self.β, self.W, self.v)
            return MatPlotLib.draw_gaussian_mixture(
                x, params, r, title="Prior distribution.", display_ids=display_ids
            )

        if d == "empirical_prior":
            params = (self.m_bar, self.β_bar, self.W_bar, self.v_bar)
            return MatPlotLib.draw_gaussian_mixture(
                x, params, r, title="Empirical prior distribution.", display_ids=display_ids
            )

        if d == "posterior":
            params = (self.m_hat, self.β_hat, self.W_hat, self.v_hat)
            return MatPlotLib.draw_gaussian_mixture(
                x, params, r, title="Posterior distribution.", display_ids=display_ids
            )

        raise RuntimeError(f"Unsupported distribution type '{d}'.")

    def draw_fixed_components(self, x, r, d="posterior"):

        counts = self.fixed_gaussian.fixed_n_steps
        if counts is not None:
            counts = [int(count.item()) for count in counts]

        if d == "prior":
            params = (self.m, self.β, self.W, self.v)
            return MatPlotLib.draw_fixed_components(
                x, self.fixed_components, params, r, title="Prior distribution.", counts=counts
            )

        if d == "empirical_prior":
            params = (self.m_bar, self.β_bar, self.W_bar, self.v_bar)
            return MatPlotLib.draw_fixed_components(
                x, self.fixed_components, params, r, title="Empirical prior distribution.", counts=counts
            )

        if d == "posterior":
            params = (self.m_hat, self.β_hat, self.W_hat, self.v_hat)
            return MatPlotLib.draw_fixed_components(
                x, self.fixed_components, params, r, title="Posterior distribution.", counts=counts
            )

        raise RuntimeError(f"Unsupported distribution type '{d}'.")

    @staticmethod
    def draw_responsibilities(r, d="posterior"):

        if d == "prior":
            return MatPlotLib.draw_histograms(r, title="Prior responsibilities.")

        if d == "empirical_prior":
            return MatPlotLib.draw_histograms(r, title="Empirical prior responsibilities.")

        if d == "posterior":
            return MatPlotLib.draw_histograms(r, title="Posterior responsibilities.")

        raise RuntimeError(f"Unsupported distribution type '{d}'.")
