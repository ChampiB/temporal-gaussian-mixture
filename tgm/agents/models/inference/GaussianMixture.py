import torch
from torch import digamma, tensor, logdet, zeros, matmul, unsqueeze, softmax, outer, inverse, mvlgamma, trace
from torch.special import gammaln
import math

from tgm.agents.datasets.Dataset import Dataset


class GaussianMixture:

    @staticmethod
    def clone(v, d, β, m, W):
        W_cloned = None if W is None else [W_k.clone() for W_k in W]
        m_cloned = None if m is None else [m_k.clone() for m_k in m]
        v_cloned = None if v is None else v.clone()
        β_cloned = None if β is None else β.clone()
        d_cloned = None if d is None else d.clone()
        return v_cloned, d_cloned, β_cloned, m_cloned, W_cloned

    @staticmethod
    def expected_log_D(d):
        return digamma(d) - digamma(d.sum())

    @staticmethod
    def expected_log_det_Λ(W_hat, v_hat):
        n_states = len(W_hat)
        log_det = []
        for k in range(n_states):
            digamma_sum = sum([digamma((v_hat[k] + 1 - i) / 2) for i in range(n_states)])
            log_det.append(n_states * math.log(2) + logdet(W_hat[k]) + digamma_sum)
        return tensor(log_det)

    @staticmethod
    def expected_quadratic_form(obs, m_hat, β_hat, W_hat, v_hat):
        dataset_size = obs.shape[0]
        n_states = len(W_hat)
        quadratic_form = zeros([dataset_size, n_states])
        for n in range(dataset_size):
            for k in range(n_states):
                diff = obs[n] - m_hat[k]
                quadratic_form[n][k] = n_states / β_hat[k]
                quadratic_form[n][k] += v_hat[k] * matmul(diff.t(), matmul(W_hat[k], diff))
        return quadratic_form

    @staticmethod
    def responsibilities(x, m_hat, β_hat, W_hat, v_hat, log_D, log_det_Λ):
        dataset_size = x.shape[0]
        expected_log_D = unsqueeze(log_D, dim=0).repeat(dataset_size, 1)
        quadratic_form = GaussianMixture.expected_quadratic_form(x, m_hat, β_hat, W_hat, v_hat)
        log_det = unsqueeze(log_det_Λ, dim=0).repeat(dataset_size, 1)
        log_ρ = expected_log_D - 0.5 * (len(W_hat) * math.log(2 * math.pi) - log_det + quadratic_form)
        return softmax(log_ρ, dim=1)

    @staticmethod
    def N(r_hat):
        if r_hat.shape[0] == 0:
            return zeros([r_hat.shape[1]])
        return r_hat.sum(dim=0)

    @staticmethod
    def x_bar(x, r_hat, N):
        if r_hat.shape[0] == 0:
            return zeros([r_hat.shape[1], x.shape[1]])
        dataset_size = r_hat.shape[0]
        n_states = r_hat.shape[1]
        x_bar = [sum(r_hat[n][k] * x[n] for n in range(dataset_size)) / N[k] for k in range(n_states)]
        return Dataset.to_tensor(x_bar)

    @staticmethod
    def S(x, r_hat, N, x_bar):
        n_states = r_hat.shape[1]
        return [GaussianMixture.S_k(x, r_hat, N, x_bar, k) for k in range(n_states)]

    @staticmethod
    def S_k(x, r_hat, N, x_bar, k):
        if r_hat.shape[0] == 0:
            return zeros([x.shape[1], x.shape[1]])
        n_observations = x_bar.shape[1]
        S = zeros([n_observations, n_observations])
        for n in range(x.shape[0]):
            diff = x[n] - x_bar[k]
            S += r_hat[n][k] * outer(diff, diff)
        return S / N[k]

    @staticmethod
    def m_hat(β, m, N, x_bar, β_hat):
        n_states = N.shape[0]
        x_bar = torch.nan_to_num(x_bar)
        return [(β[k] * m[k] + N[k] * x_bar[k]) / β_hat[k] for k in range(n_states)]

    @staticmethod
    def W_hat(W, N, S, x_bar, m, β, β_hat):
        n_states = len(W)
        x_bar = torch.nan_to_num(x_bar)
        S = [torch.nan_to_num(S[k]) for k in range(n_states)]
        return [GaussianMixture.W_hat_k(W, N, S, x_bar, m, β, β_hat, k) for k in range(n_states)]

    @staticmethod
    def W_hat_k(W, N, S, x_bar, m, β, β_hat, k):
        W_hat = inverse(W[k]) + N[k] * S[k]
        x = x_bar[k] - m[k]
        W_hat += (β[k] * N[k] / β_hat[k]) * outer(x, x)
        return inverse(W_hat)

    @staticmethod
    def vfe(gm):

        # Pre-compute useful terms.
        log_β_hat = gm.β_hat.log()
        log_β = gm.β.log()
        ln2pi = math.log(2 * math.pi)
        ln2 = math.log(2)
        n_states = len(gm.W_hat)

        # Add part of E[P(D)] and E[Q(D)].
        F = - GaussianMixture.beta_ln(gm.d_hat) + GaussianMixture.beta_ln(gm.d)

        for k in range(n_states):

            # Add E[P(Z|D)], as well as part of E[P(D)] and E[Q(D)].
            F += (gm.d_hat[k] - gm.d[k] - gm.N[k]) * gm.log_D[k]

            # Add E[Q(μ|Λ)].
            F += 0.5 * (n_states * (log_β_hat[k] - ln2pi) + gm.log_det_Λ[k] - n_states)

            # Add E[P(μ|Λ)].
            diff = gm.m[k] - gm.m_hat[k]
            quadratic_form = (
                    n_states * gm.β[k] / gm.β_hat[k] +
                    gm.β[k] * gm.v_hat[k] * matmul(matmul(diff.t(), gm.W_hat[k]), diff)
            )
            F -= 0.5 * (n_states * (log_β[k] - ln2pi) + gm.log_det_Λ[k] - quadratic_form)

            # Add E[Q(Λ)].
            F += -0.5 * (
                    gm.v_hat[k] * n_states * ln2 + gm.v_hat[k] * logdet(gm.W_hat[k]) -
                    (gm.v_hat[k] - n_states - 1) * gm.log_det_Λ[k] + gm.v_hat[k] * n_states
            ) - mvlgamma(gm.v_hat[k] / 2, n_states)

            # Add E[P(Λ)].
            F += 0.5 * (
                    gm.v[k] * n_states * ln2 + gm.v[k] * logdet(gm.W[k]) -
                    (gm.v[k] - n_states - 1) * gm.log_det_Λ[k] +
                    gm.v_hat[k] * trace(matmul(inverse(gm.W[k]), gm.W_hat[k]))
            ) + mvlgamma(gm.v[k] / 2, n_states)

            # Add E[P(X|Z,μ,Λ)].
            if gm.N[k] != 0:
                diff = gm.x_bar[k] - gm.m_hat[k]
                F -= 0.5 * gm.N[k] * (
                        gm.log_det_Λ[k] - n_states / gm.β_hat[k] -
                        gm.v_hat[k] * trace(matmul(gm.S[k], gm.W_hat[k])) -
                        gm.v_hat[k] * matmul(matmul(diff.t(), gm.W_hat[k]), diff) - n_states * ln2pi
                )

        # Add E[Q(Z)].
        log_r_hat = torch.nan_to_num(gm.r_hat.log())
        F += (gm.r_hat * log_r_hat).sum()
        return F

    @staticmethod
    def beta_ln(x):
        beta_ln = 0
        for i in range(x.shape[0]):
            beta_ln += gammaln(x[i])
        beta_ln -= gammaln(x.sum())
        return beta_ln
