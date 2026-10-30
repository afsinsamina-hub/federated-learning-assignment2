"""
Differential-privacy tools with formal accounting.

  * clip_l2 / gaussian_noise : Gaussian mechanism on a clipped model update (sensitivity C).
  * RdpAccountant            : Renyi-DP accountant for the (Poisson-)sampled Gaussian mechanism
                               (Mironov, Talwar & Zhang 2019, integer orders), converted to (eps, delta)-DP
                               with the tight conversion of Balle et al. (2020).
  * rappor_perturb / rappor_epsilon : basic RAPPOR local DP of Erlingsson et al. (CCS 2014):
                               every bit is kept with probability p and flipped with q = 1 - p;
                               the mechanism is eps-LDP with eps = 2 ln(p/q).

Accounting assumptions (stated in the report): clients are sampled with probability q = m/K per round
(the simulation samples exactly m of K, which the Poisson bound approximates), the clipped update of a
client is the unit of privacy (user-level DP), and the noise multiplier z = sigma / C.
"""
import math
import numpy as np


def clip_l2(u, C):
    n = np.linalg.norm(u)
    return u * min(1.0, C / (n + 1e-12))


def gaussian_noise(shape, sigma, rng):
    return rng.normal(0.0, sigma, size=shape)


def _log_add(a, b):
    m = max(a, b)
    return m if m == -math.inf else m + math.log(math.exp(a - m) + math.exp(b - m))


def rdp_sampled_gaussian(q, z, alpha):
    """RDP of one step of the sampled Gaussian mechanism at integer order alpha >= 2."""
    if q == 0: return 0.0
    if q == 1.0: return alpha / (2 * z ** 2)
    log_a = -math.inf
    for k in range(alpha + 1):
        log_coef = (math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
                    + (alpha - k) * math.log(1 - q) + k * math.log(q))
        log_a = _log_add(log_a, log_coef + (k * k - k) / (2 * z ** 2))
    return log_a / (alpha - 1)


ORDERS = list(range(2, 257))


class RdpAccountant:
    def __init__(self, q, noise_multiplier):
        self.q, self.z, self.steps = q, noise_multiplier, 0
        self._rdp1 = np.array([rdp_sampled_gaussian(q, noise_multiplier, a) for a in ORDERS])

    def step(self, n=1): self.steps += n

    def epsilon(self, delta, tight=True):
        rdp = self.steps * self._rdp1
        best = math.inf
        for a, r in zip(ORDERS, rdp):
            if tight:      # Balle et al. 2020, Thm 21 (as used in Opacus / TF-Privacy >= 0.8)
                e = r + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
            else:          # classic conversion of Mironov 2017
                e = r + math.log(1 / delta) / (a - 1)
            best = min(best, e)
        return max(best, 0.0)


def rappor_epsilon(p):
    q = 1 - p
    return 2 * math.log(p / q)


def rappor_p_from_epsilon(eps):
    return math.exp(eps / 2) / (1 + math.exp(eps / 2))


def rappor_perturb(bits, p, rng):
    """Basic RAPPOR: keep each bit with prob. p, flip with prob. 1-p."""
    flip = rng.random(bits.shape) >= p
    return np.where(flip, 1 - bits, bits)
