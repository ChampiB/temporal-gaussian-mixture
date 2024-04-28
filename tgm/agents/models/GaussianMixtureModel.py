from copy import deepcopy

from tgm.agents.datasets.Dataset import Dataset
from tgm.agents.models.display.MatPlotLib import MatPlotLib
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
        self.fixed_components = []

        # Pre-computed terms.
        self.log_D = self.log_det_Λ = self.N = self.x_bar = self.S = self.vfe = None
        self.N_prime = self.x_prime = self.S_prime = None
        self.N_second = self.x_second = self.S_second = None

    def fit(self, x_forget, x_keep, split=True, threshold=1, debugger=None):

        n_steps = 0
        while n_steps < 5:

            # Keep track of previous variational free energy.
            vfe = self.vfe

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
                (self.N_prime[k] * self.S_prime[k] + self.N_second[k] * self.S_second[k]) / self.N[k]
                for k in range(len(self.S_prime))
            ]
            self.x_bar = [
                (self.N_prime[k] * self.x_prime[k] + self.N_second[k] * self.x_second[k]) / self.N[k]
                for k in range(len(self.S))
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

        # Try splitting the Gaussian components that are not fixed yet.
        if split is True:
            self.split_flexible_components(x_keep)

    def split_flexible_components(self, x_keep):
        pass  # TODO

    def is_initialized(self):
        return self.W is not None

    def initialize(self, x, init_type="mean-shift"):

        # Initialize the prior parameters and the responsibilities using the k-means algorithm.
        init_fc = {
            "k-means": KMeans.init_gm,
            "mean-shift": MeanShift.init_gm
        }
        self.v, self.d, self.β, self.m, self.W, self.r_hat = init_fc[init_type](x, self.n_observations, self.n_states)

        # Initialize the empirical prior parameters.
        self.v_bar, self.d_bar, self.β_bar, self.m_bar, self.W_bar = GMix.clone(self.v, self.d, self.β, self.m, self.W)

        # Initialize the posterior parameters.
        self.v_hat, self.d_hat, self.β_hat, self.m_hat, self.W_hat = GMix.clone(self.v, self.d, self.β, self.m, self.W)

        # Initialize pre-computed terms.
        self.log_D = GMix.expected_log_D(self.d_hat)
        self.log_det_Λ = GMix.expected_log_det_Λ(self.W_hat, self.v_hat)
        self.N = GMix.N(self.r_hat)
        self.x_bar = GMix.x_bar(x, self.r_hat, self.N)
        self.S = GMix.S(x, self.r_hat, self.N, self.x_bar)
        self.vfe = GMix.vfe(self)

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
        gm.fixed_components = deepcopy(self.fixed_components)

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

    def draw_distribution(self, x, r, d="posterior"):

        if d == "prior":
            params = (self.m, self.β, self.W, self.v)
            return MatPlotLib.draw_gaussian_mixture(x, params, r, title="Prior distribution.")

        if d == "empirical_prior":
            params = (self.m_bar, self.β_bar, self.W_bar, self.v_bar)
            return MatPlotLib.draw_gaussian_mixture(x, params, r, title="Empirical prior distribution.")

        if d == "posterior":
            params = (self.m_hat, self.β_hat, self.W_hat, self.v_hat)
            return MatPlotLib.draw_gaussian_mixture(x, params, r, title="Posterior distribution.")

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

