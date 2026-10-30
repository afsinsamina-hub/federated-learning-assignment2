"""
Model architectures.

  MLP      : baseline classifier used by FedAvg (and PEFL/BPPFL when their papers use a plain network).
  RBM/EDBN : Entropy Deep Belief Network of Bhardwaj & Sumangali (2025): stacked Bernoulli RBMs whose hidden-layer
             widths obey the information-entropy bounds N_min <= N_hid <= N_in (Eqs. 33-42), followed by a
             logistic output layer. RBMs are pre-trained with contrastive divergence (CD-1), then the whole
             stack is fine-tuned with back-propagation.
  TabNet   : the original TabNet architecture of Arik & Pfister (AAAI 2021) -- attentive transformer with sparsemax
             masks, shared + step-specific GLU feature transformers, ghost batch-norm, relaxation factor gamma --
             taken from the reference PyTorch implementation (pytorch-tabnet, Dreamquark).
"""
import math
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)


class MLP(nn.Module):
    def __init__(self, d, hidden=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.reg = 0.0

    def forward(self, x): return self.net(x).squeeze(-1)


class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_vis, n_hid) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_vis))     # visible bias
        self.b = nn.Parameter(torch.zeros(n_hid))     # hidden bias

    def h_prob(self, v): return torch.sigmoid(v @ self.W + self.b)
    def v_prob(self, h): return torch.sigmoid(h @ self.W.t() + self.a)

    def cd1_step(self, v, lr, rng_t):
        """One contrastive-divergence (CD-1) update; returns the reconstruction error."""
        with torch.no_grad():
            ph = self.h_prob(v); h = (torch.rand(ph.shape, generator=rng_t) < ph).float()
            pv = self.v_prob(h); ph2 = self.h_prob(pv)
            n = v.shape[0]
            self.W += lr * (v.t() @ ph - pv.t() @ ph2) / n
            self.a += lr * (v - pv).mean(0)
            self.b += lr * (ph - ph2).mean(0)
            return float(((v - pv) ** 2).mean())


def binary_entropy_bits(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def entropy_lower_bound(X):
    """Eq. (36)-(40): information entropy of the visible layer H_vis = sum_i H(p_i);
    the hidden layer (each node carries <= 1 bit) needs N_hid >= H_vis."""
    return int(math.ceil(binary_entropy_bits(X.mean(0)).sum()))


class EDBN(nn.Module):
    """Depth D (number of RBMs) with N_hid = N_in nodes per hidden layer (Eq. 41), checked against the
    entropy lower bound; logistic output layer for the binary task."""
    def __init__(self, d, depth=2):
        super().__init__()
        self.rbms = nn.ModuleList([RBM(d, d) for _ in range(depth)])
        self.out = nn.Linear(d, 1)
        self.reg = 0.0

    def forward(self, x):
        for r in self.rbms: x = r.h_prob(x)
        return self.out(x).squeeze(-1)


class TabNetModel(nn.Module):
    def __init__(self, d, n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2, n_shared=2, lambda_sparse=1e-3):
        super().__init__()
        from pytorch_tabnet.tab_network import TabNet
        from pytorch_tabnet.utils import create_group_matrix
        self.net = TabNet(input_dim=d, output_dim=1, n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
                          cat_idxs=[], cat_dims=[], cat_emb_dim=[], n_independent=n_independent, n_shared=n_shared,
                          virtual_batch_size=128, momentum=0.02, mask_type="sparsemax",
                          group_attention_matrix=create_group_matrix([], d))
        self.lambda_sparse = lambda_sparse
        self.reg = 0.0

    def forward(self, x):
        out, m_loss = self.net(x)
        self.reg = -self.lambda_sparse * m_loss        # sparsity regulariser of TabNet (entropy of the masks)
        return out.squeeze(-1)

    def masks(self, x):
        """Feature-selection masks of the decision steps (used for the TabNet explanation)."""
        return self.net.forward_masks(x)


# -------------------------------------------------------- flat state-vector helpers (parameters + float buffers)
def state_keys(model):
    sd = model.state_dict()
    return [k for k, v in sd.items() if v.dtype.is_floating_point]


def get_vec(model):
    sd = model.state_dict()
    return torch.cat([sd[k].reshape(-1).double() for k in state_keys(model)]).numpy().copy()


def set_vec(model, vec):
    sd = model.state_dict(); i = 0
    t = torch.from_numpy(np.asarray(vec))
    for k in state_keys(model):
        n = sd[k].numel(); sd[k].copy_(t[i:i + n].reshape(sd[k].shape).to(sd[k].dtype)); i += n
    for mod in model.modules():                      # noisy aggregation can push a variance below 0 -> keep valid
        if isinstance(mod, nn.modules.batchnorm._BatchNorm) and mod.running_var is not None:
            mod.running_var.clamp_(min=1e-5)


def build(name, d, seed):
    torch.manual_seed(seed)
    return {"mlp": lambda: MLP(d), "edbn": lambda: EDBN(d), "tabnet": lambda: TabNetModel(d)}[name]()
