"""
Poisoning attacks used in the PPFBXAIO paper (extra noise, label flipping, STAT-OPT, DYN-OPT).

STAT-OPT and DYN-OPT are the optimisation attacks of Shejwalkar & Houmansadr (NDSS 2021),
"Manipulating the Byzantine". A malicious update has the form
        u' = u_b + gamma * p ,         u_b = mean of the benign updates seen by the attacker
with a perturbation direction p and the largest gamma that keeps u' inside the cloud of benign updates
(Min-Max criterion):  max_i ||u' - u_i||  <=  max_{i,j} ||u_i - u_j||   (i, j over benign updates).

  STAT-OPT : static direction p = -sign(u_b) (unit-norm scaled); gamma is optimised ONCE (first attacked
             round) and then kept fixed for the rest of training.
  DYN-OPT  : dynamic direction p = -std_i(u_i) (coordinate-wise std of the benign updates);
             gamma is re-optimised in every round with the Min-Max criterion.
  noise    : u' = u + N(0, s^2 I) with s chosen so that the noise norm is `noise_mult` times the mean
             benign update norm ("noise far beyond the privacy budget").
  label_flip : the client trains on flipped labels (applied to the data, see engine).
"""
import numpy as np


def _minmax_threshold(B):
    d = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=2)
    return d.max()


def optimise_gamma(u_b, p, B, hi=1e3, iters=40):
    """Largest gamma with max_i ||u_b + gamma p - u_i|| <= max_ij ||u_i - u_j|| (bisection)."""
    thr = _minmax_threshold(B)
    ok = lambda g: np.linalg.norm((u_b + g * p)[None, :] - B, axis=1).max() <= thr
    lo = 0.0
    if not ok(1e-9): return 0.0
    while ok(hi) and hi < 1e12: hi *= 10
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


class OptAttack:
    def __init__(self, kind):
        assert kind in ("stat_opt", "dyn_opt"); self.kind, self.gamma = kind, None

    def craft(self, benign):
        """benign: (n, P) array of benign updates seen in this round. Returns one malicious update."""
        B = np.asarray(benign)
        if len(B) == 1:                       # a single benign update: no spread to hide in -> use 0.5 norm of it
            return -0.5 * B[0]
        u_b = B.mean(0)
        if self.kind == "stat_opt":
            p = -np.sign(u_b); p = p / (np.linalg.norm(p) + 1e-12) * np.linalg.norm(u_b)
            if self.gamma is None: self.gamma = optimise_gamma(u_b, p, B)       # static: fixed after round 1
            g = self.gamma
        else:
            p = -B.std(0)
            g = optimise_gamma(u_b, p, B)                                         # dynamic: every round
        return u_b + g * p


def noise_attack(u, benign_norm, noise_mult, rng):
    s = noise_mult * benign_norm / np.sqrt(len(u))
    return u + rng.normal(0.0, s, size=u.shape)
