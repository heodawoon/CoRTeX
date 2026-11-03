import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRFPositionalEncoding(nn.Module):
    """
    Mildenhall, Ben, et al. "Nerf: Representing scenes as neural radiance fields for view synthesis." Communications of the ACM 65.1 (2021): 99-106.
    """
    def __init__(self, n_freqs=6):
        super().__init__()
        self.n_freqs = n_freqs
        freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

    @property
    def output_dim(self):
        return 6 * self.n_freqs

    def forward(self, coords):
        scaled = coords.unsqueeze(-1) * self.freqs.view(1, 1, -1) * math.pi
        sin_enc = torch.sin(scaled)
        cos_enc = torch.cos(scaled)
        enc = torch.stack([sin_enc, cos_enc], dim=-1)
        return enc.flatten(start_dim=1)


class HebbianModule(nn.Module):
    """
    Ba, Jimmy, et al. "Using fast weights to attend to the recent past." Advances in neural information processing systems 29 (2016).
    """
    def __init__(self, R, alpha_init=0.5, rate_init_std=0.5, H_clip=5.0,
                 gamma_init_bias=0.0, eta_init_bias=0.0):
        super().__init__()
        self.R = R
        self.H_clip = H_clip
        self.eta_logits   = nn.Parameter(torch.randn(R) * rate_init_std + eta_init_bias)
        self.gamma_logits = nn.Parameter(torch.randn(R) * rate_init_std + gamma_init_bias)
        self.alpha        = nn.Parameter(torch.tensor(float(alpha_init)))

    def update(self, S_t, H_prev):
        gamma = torch.sigmoid(self.gamma_logits).view(1, -1, 1)
        eta   = F.softplus(self.eta_logits).view(1, -1, 1)
        H_new = gamma * H_prev + eta * S_t
        H_new = H_new.clamp(min=-self.H_clip, max=self.H_clip)
        bias  = self.alpha * H_new
        return H_new, bias

    @torch.no_grad()
    def rates(self):
        return {
            'eta':   F.softplus(self.eta_logits).detach().cpu(),
            'gamma': torch.sigmoid(self.gamma_logits).detach().cpu(),
            'alpha': self.alpha.detach().cpu(),
        }


def _init_rsti_weights(R, d_hidden, M_max):
    scale_w1 = (2.0 / M_max) ** 0.5
    scale_w2 = (1.0 / d_hidden) ** 0.5
    W_1 = nn.Parameter(torch.randn(R, d_hidden, M_max) * scale_w1)
    b_1 = nn.Parameter(torch.zeros(R, d_hidden))
    W_2 = nn.Parameter(torch.randn(R, d_hidden) * scale_w2)
    b_2 = nn.Parameter(torch.zeros(R))
    return W_1, b_1, W_2, b_2


class RSTI(nn.Module):
    def __init__(self, R, M_max, d_hidden=1024, dropout=0.0,
                 sharpness=10.0, init_logits_std=1.0, diversity_weight=0.01):
        super().__init__()
        self.R = R
        self.M_max = M_max
        self.M = M_max
        self.d_hidden = d_hidden
        self.sharpness = sharpness
        self.diversity_weight = diversity_weight

        self.M_logits = nn.Parameter(torch.randn(R) * init_logits_std)
        self.W_1, self.b_1, self.W_2, self.b_2 = _init_rsti_weights(R, d_hidden, M_max)
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(M_max).float()
        self.register_buffer('_distance_from_newest', (M_max - 1) - positions)

    def get_M_r(self):
        return self.M_max * torch.sigmoid(self.M_logits)

    def get_mask(self):
        M_r = self.get_M_r().unsqueeze(-1)
        d = self._distance_from_newest.unsqueeze(0)
        return torch.sigmoid(self.sharpness * (M_r - d - 0.5))

    def diversity_loss(self):
        return -self.diversity_weight * self.get_M_r().var()

    def forward(self, A):
        A = A * self.get_mask().unsqueeze(0)
        h = torch.einsum('brm, rhm -> brh', A, self.W_1) + self.b_1
        h = F.gelu(h)
        h = self.dropout(h)
        return torch.einsum('brh, rh -> br', h, self.W_2) + self.b_2


class PerRegionProjection(nn.Module):
    def __init__(self, R, d_hidden=16, dropout=0.3):
        super().__init__()
        self.R = R
        self.d_hidden = d_hidden
        self.W_1 = nn.Parameter(torch.randn(R, 2, d_hidden) * (2.0 / 2) ** 0.5 * 0.5)
        self.b_1 = nn.Parameter(torch.zeros(R, d_hidden))
        self.W_2 = nn.Parameter(torch.randn(R, d_hidden) * (1.0 / d_hidden) ** 0.5)
        self.b_2 = nn.Parameter(torch.zeros(R))
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_prev, x_tilde_t):
        combined = torch.stack([z_prev, x_tilde_t], dim=-1)
        h = torch.einsum('bri, rih -> brh', combined, self.W_1) + self.b_1
        h = F.gelu(h)
        h = self.dropout(h)
        u_t = torch.einsum('brh, rh -> br', h, self.W_2) + self.b_2
        return u_t
