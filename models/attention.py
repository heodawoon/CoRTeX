import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_modules import NeRFPositionalEncoding


class IdentityAwareAttention(nn.Module):
    POSITION_MODES = ('none', 'one_hot', 'nerf', 'nerf_plus_onehot')
    ATTN_KINDS     = ('standard', 'hebbian')
    GATE_MODES = ('gate_bold_ctx', 'ctx_only', 'sum', 'mean')

    def __init__(self, R, d_attn=64, position_mode='nerf_plus_onehot',
                 attn_kind='standard', nerf_n_freqs=6,
                 include_bold_in_feat=True, gate_mode='gate_bold_ctx'):
        super().__init__()
        assert position_mode in self.POSITION_MODES
        assert attn_kind     in self.ATTN_KINDS
        assert gate_mode     in self.GATE_MODES
        self.R = R
        self.d_attn = d_attn
        self.position_mode = position_mode
        self.attn_kind     = attn_kind
        self.include_bold_in_feat = include_bold_in_feat
        self.gate_mode = gate_mode

        if position_mode in ('one_hot', 'nerf_plus_onehot'):
            self.register_buffer('one_hot', torch.eye(R))
        self.nerf_encoder = None
        if position_mode in ('nerf', 'nerf_plus_onehot'):
            self.nerf_encoder = NeRFPositionalEncoding(n_freqs=nerf_n_freqs)

        feat_dim = R
        if include_bold_in_feat:
            feat_dim += 1
        if position_mode in ('one_hot', 'nerf_plus_onehot'):
            feat_dim += R
        if position_mode in ('nerf', 'nerf_plus_onehot'):
            feat_dim += self.nerf_encoder.output_dim
        self.feat_dim = feat_dim

        self.attn_norm = nn.LayerNorm(R)
        self.W_Q = nn.Linear(feat_dim, d_attn, bias=False)
        self.W_K = nn.Linear(feat_dim, d_attn, bias=False)

        self.roi_gate_logits = nn.Parameter(torch.zeros(R))

        self._centroids_set = (self.nerf_encoder is None)

    def set_centroids(self, centroids):
        assert self.nerf_encoder is not None
        centroids = centroids.to(dtype=torch.float32)
        self.register_buffer('centroids', centroids)
        with torch.no_grad():
            enc = self.nerf_encoder(centroids)
        self.register_buffer('nerf_encoding_cached', enc)
        self._centroids_set = True

    def build_features(self, S_n, bold_t):
        B = S_n.shape[0]
        parts = [S_n]
        if self.position_mode in ('one_hot', 'nerf_plus_onehot'):
            parts.append(self.one_hot.unsqueeze(0).expand(B, -1, -1))
        if self.position_mode in ('nerf', 'nerf_plus_onehot'):
            parts.append(self.nerf_encoding_cached.unsqueeze(0).expand(B, -1, -1))
        if self.include_bold_in_feat:
            parts.append(bold_t.unsqueeze(-1))
        return torch.cat(parts, dim=-1)

    def forward(self, S_t, bold_t, hebbian_bias=None):
        S_n = self.attn_norm(S_t)
        f_t = self.build_features(S_n, bold_t)
        Q_t = self.W_Q(f_t)
        K_t = self.W_K(f_t)
        logits = torch.einsum('brd, bsd -> brs', Q_t, K_t) / (self.d_attn ** 0.5)

        if self.attn_kind == 'hebbian' and hebbian_bias is not None:
            logits = logits + hebbian_bias

        attn      = F.softmax(logits, dim=-1)
        cross_ctx = torch.einsum('brs, bs -> br', attn, bold_t)

        gate = torch.sigmoid(self.roi_gate_logits)
        if self.gate_mode == 'gate_bold_ctx':
            x_tilde_t = gate.unsqueeze(0) * bold_t + (1.0 - gate.unsqueeze(0)) * cross_ctx
        elif self.gate_mode == 'ctx_only':
            x_tilde_t = cross_ctx
        elif self.gate_mode == 'sum':
            x_tilde_t = bold_t + cross_ctx
        elif self.gate_mode == 'mean':
            x_tilde_t = 0.5 * (bold_t + cross_ctx)
        else:
            x_tilde_t = gate.unsqueeze(0) * bold_t + (1.0 - gate.unsqueeze(0)) * cross_ctx
        return x_tilde_t, attn, gate
