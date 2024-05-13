import torch

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

        # The parameters of the fixed components (only set during initialization, for debugging purpose).
        self.W_fixed = self.m_fixed = self.v_fixed = self.β_fixed = self.d_fixed = None

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

    def fit(self, x_forget, x_keep, debugger, split=True, threshold=1):

        n_steps = 0
        i = 0
        while n_steps < 5 and i < 100:

            # Notify the debugger that a step of variational inference and a Z update are starting.
            debugger.before("vi_step", auto_index=True)
            debugger.before("update_Z", new_checkpoint=False)

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

            # Notify the debugger that a Z update is ending and that a D update is starting.
            debugger.after("update_Z")
            debugger.before("update_D", new_checkpoint=False)

            # Update for D.
            self.d_bar = self.d + self.N_prime
            self.d_hat = self.d_bar + self.N_second
            self.log_D = GMix.expected_log_D(self.d_hat)

            # Notify the debugger that a D update is ending and that a μ and Λ update is starting.
            debugger.after("update_D")
            debugger.before("update_μ_and_Λ", new_checkpoint=False)

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

            # Update variational free energy and the number of steps.
            self.vfe = GMix.vfe(self)
            n_steps = n_steps + 1 if float(vfe - self.vfe) < threshold else 0

            # Notify the debugger that a μ and Λ update, as well as a step of variational inference are ending.
            debugger.after("update_μ_and_Λ")
            debugger.after("vi_step", new_checkpoint=False)

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

    def initialize(self, x, debugger, init_type="mean-shift"):

        # Retrieve the parameters of the fixed components.
        self.v_fixed, self.d_fixed, self.β_fixed, self.m_fixed, self.W_fixed = self.parameters_of(self.fixed_components)

        # Add checkpoint from which fixed component parameters can be accessed.
        debugger.before("prior_initialization", auto_index=True)

        # Compute all the component parameters.
        init_fc = {
            "k-means": KMeans.init_gm,
            "mean-shift": MeanShift.init_gm
        }
        self.v, self.d, self.β, self.m, self.W, self.r_hat = \
            init_fc[init_type](x, self.n_observations, self.n_states)

        # Retrieve the parameters of the flexible components.
        fixed_components = self.fixed_gaussian.new_indices_of(
            self.v_fixed, self.m_fixed, self.W_fixed, self.v, self.m, self.W
        )
        flexible_components = [k for k in range(len(self.m)) if k not in fixed_components]
        self.v, self.d, self.β, self.m, self.W = self.parameters_of(flexible_components)

        # Add checkpoint from which flexible component parameters can be accessed.
        debugger.middle("prior_initialization")

        # Merge the parameters of the fixed and flexible components.
        if self.v_fixed is not None:
            self.v, self.d, self.β, self.m, self.W = self.merge(
                self.v_fixed, self.d_fixed, self.β_fixed, self.m_fixed, self.W_fixed,
                self.v, self.d, self.β, self.m, self.W
            )

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

        # Add checkpoint from which all component parameters can be accessed.
        debugger.after("prior_initialization")

    @staticmethod
    def merge(v0, d0, β0, m0, W0, v1, d1, β1, m1, W1):

        # If any set of parameters are None, return the other set.
        if v1 is None or d1 is None or β1 is None or m1 is None or W1 is None:
            return v0, d0, β0, m0, W0
        if v0 is None or d0 is None or β0 is None or m0 is None or W0 is None:
            return v1, d1, β1, m1, W1

        # Otherwise, merge the two sets of parameters.
        v = torch.concat([v0, v1], dim=0)
        d = torch.concat([d0, d1], dim=0)
        β = torch.concat([β0, β1], dim=0)
        m = m0 + m1
        W = W0 + W1
        return v, d, β, m, W

    def parameters_of(self, components, d="prior"):

        if len(components) == 0:
            return None, None, None, None, None

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
        n_states = len(components)
        extracted_v = torch.zeros([n_states])
        extracted_d = torch.zeros([n_states])
        extracted_β = torch.zeros([n_states])
        extracted_m = []
        extracted_W = []
        for j, component in enumerate(components):
            extracted_v[j] = v[component]
            extracted_d[j] = d[component]
            extracted_β[j] = β[component]
            extracted_m.append(m[component].clone())
            extracted_W.append(W[component].clone())

        return extracted_v, extracted_d, extracted_β, extracted_m, extracted_W

    def update_fixed_components(self, debugger):
        debugger.before("update_fixed_components", auto_index=True)
        self.fixed_gaussian.update(self)
        debugger.after("update_fixed_components")

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
        return [i for i, N_k in enumerate(self.N) if N_k != 0.0]

    def update_prior_parameters(self):

        # Update the prior parameters using the empirical prior parameters.
        self.v, self.d, self.β, self.m, self.W = GMix.clone(self.v_bar, self.d_bar, self.β_bar, self.m_bar, self.W_bar)

    def clone(self):
        
        # Create a new Gaussian mixture.
        gm = GaussianMixtureModel(self.n_states, self.n_observations)

        # Clone the parameter of the fixed components.
        gm.v_fixed, gm.d_fixed, gm.β_fixed, gm.m_fixed, gm.W_fixed = \
            GMix.clone(self.v_fixed, self.d_fixed, self.β_fixed, self.m_fixed, self.W_fixed)

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
        gm.log_D = None if self.log_D is None else self.log_D.clone()
        gm.log_det_Λ = None if self.log_det_Λ is None else self.log_det_Λ.clone()
        gm.N = None if self.N is None else self.N.clone()
        gm.x_bar = None if self.x_bar is None else self.x_bar.clone()
        gm.S = None if self.S is None else [S_k.clone() for S_k in self.S]
        gm.N_prime = None if self.N_prime is None else self.N_prime.clone()
        gm.x_prime = None if self.x_prime is None else self.x_prime.clone()
        gm.S_prime = None if self.S_prime is None else [S_k.clone() for S_k in self.S_prime]
        gm.N_second = None if self.N_second is None else self.N_second.clone()
        gm.x_second = None if self.x_second is None else self.x_second.clone()
        gm.S_second = None if self.S_second is None else [S_k.clone() for S_k in self.S_second]

        # Clone the variational free energy, if possible.
        try:
            gm.vfe = None if self.β_hat is None else GMix.vfe(self)
        except (IndexError, AttributeError):
            gm.vfe = None

        return gm

    def draw_distribution(self, x, r, d="posterior", display_ids=True, ellipses=True, datum=None):

        if d == "prior":
            params = (self.m, self.β, self.W, self.v)
            return MatPlotLib.draw_gaussian_mixture(
                x, params, r,
                title="Prior distribution.", display_ids=display_ids, ellipses=ellipses, datum=datum
            )

        if d == "empirical_prior":
            params = (self.m_bar, self.β_bar, self.W_bar, self.v_bar)
            return MatPlotLib.draw_gaussian_mixture(
                x, params, r,
                title="Empirical prior distribution.", display_ids=display_ids, ellipses=ellipses, datum=datum
            )

        if d == "posterior":
            params = (self.m_hat, self.β_hat, self.W_hat, self.v_hat)
            return MatPlotLib.draw_gaussian_mixture(
                x, params, r,
                title="Posterior distribution.", display_ids=display_ids, ellipses=ellipses, datum=datum
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
