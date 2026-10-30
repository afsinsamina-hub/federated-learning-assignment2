"""
Levy Grasshopper Optimisation Algorithm (LGOA) for feature selection,
following Algorithm 1 and Eqs. (3)-(15) of Bhardwaj & Sumangali, Sci. Rep. 15:21799 (2025), which build on the
Grasshopper Optimisation Algorithm (Saremi, Mirjalili & Lewis, Adv. Eng. Softw. 2017) and Levy flights
(Mantegna's algorithm).

  position X_i in [0,1]^d  ->  feature mask  (X_i > 0.5);   fitness = a*(CV error of kNN) + (1-a)*(#selected/d)
  social force      S_i = sum_{j != i} c * (ub-lb)/2 * s(|x_j - x_i|) * (x_j - x_i)/d_ij,  s(r) = f e^{-r/l} - e^{-r}
  position update   X_i <- c * S_i + T   (T = best position found so far),  then + stepsize  (Levy step)
  c = c_max - t (c_max - c_min)/L,  c_max = 1, c_min = 1e-5          (Eq. 9)
  Levy step         Mantegna: s = U / |V|^(1/beta), U~N(0, sigma_u^2), V~N(0,1), beta = 1.5   (Eqs. 11-14)
  stepsize          = 0.01 * s                                          (Eq. 15)
  initialisation    first grasshopper random, the others from the Levy distribution               (Alg. 1)
"""
import math
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def levy_step(shape, rng, beta=1.5):
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
               (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma_u, shape); v = rng.normal(0, 1, shape)
    return u / np.abs(v) ** (1 / beta)


def _s(r, f=0.5, l=1.5):
    return f * np.exp(-r / l) - np.exp(-r)


def fitness(mask, X, y, alpha=0.99, seed=0):
    if mask.sum() == 0: return 1.0
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    acc = cross_val_score(KNeighborsClassifier(5), X[:, mask], y, cv=skf).mean()
    return alpha * (1 - acc) + (1 - alpha) * mask.sum() / X.shape[1]


def lgoa_select(X, y, n_grasshoppers=10, iters=15, rng=None, alpha=0.99, c_max=1.0, c_min=1e-5, thr=0.5):
    rng = rng or np.random.default_rng(0)
    d = X.shape[1]; lb, ub = 0.0, 1.0
    pos = np.empty((n_grasshoppers, d))
    pos[0] = rng.random(d)                                           # first grasshopper: random
    pos[1:] = np.clip(0.5 + 0.1 * levy_step((n_grasshoppers - 1, d), rng), lb, ub)   # others: Levy distribution
    fit = np.array([fitness(p > thr, X, y, alpha, seed=i) for i, p in enumerate(pos)])
    best = fit.argmin(); T, Tf = pos[best].copy(), fit[best]
    for t in range(1, iters + 1):
        c = c_max - t * (c_max - c_min) / iters
        new = np.empty_like(pos)
        for i in range(n_grasshoppers):
            diff = pos - pos[i]; dist = np.linalg.norm(diff, axis=1)
            dist_n = 2 + np.remainder(dist, 2)                       # GOA maps distances into [1,3]
            S = np.zeros(d)
            for j in range(n_grasshoppers):
                if j == i: continue
                S += c * (ub - lb) / 2 * _s(dist_n[j]) * diff[j] / (dist[j] + 1e-12)
            x = c * S + T                                            # Eq. (8): wind/gravity terms dropped
            x = x + 0.01 * levy_step(d, rng) * (ub - lb)             # Eq. (15) Levy step, f = 0.01
            new[i] = np.clip(x, lb, ub)
        pos = new
        fit = np.array([fitness(p > thr, X, y, alpha, seed=i) for i, p in enumerate(pos)])
        if fit.min() < Tf: Tf, T = fit.min(), pos[fit.argmin()].copy()
    mask = T > thr
    if mask.sum() == 0: mask[np.argmax(T)] = True
    return mask, Tf
