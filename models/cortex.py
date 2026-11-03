from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import IdentityAwareAttention
from .custom_modules import HebbianModule, PerRegionProjection, RSTI
from .heads import BrainNetCNNHead, MLPHead


@dataclass
class CORTEXConfig:
    R: int = 360
    T: int = 6
    M: int = 6
    position_mode: str = 'nerf_plus_onehot'
    attn_kind:     str = 'hebbian'
    head:          str = 'brainnet'
    d_attn: int = 64
    nerf_n_freqs: int = 6
    include_bold_in_feat: bool = True
    prp_d_hidden: int = 16
    d_hidden_rsr:     int = 1024
    rsti_sharpness:        float = 10.0
    rsti_init_logits_std:  float = 1.0
    rsti_diversity_weight: float = 0.01
    hebbian_alpha_init:    float = 0.5
    hebbian_rate_init_std: float = 0.5
    hebbian_H_clip:        float = 5.0
    n_classes:       int = 12
    cls_e2e_filters: int = 8
    cls_e2n_filters: int = 16
    cls_fc_hidden:   int = 64
    mlp_head_hidden: tuple = (1024, 256)
    mlp_head_upper_tri: bool = True
    mlp_head_activation: str = 'relu'
    prp_dropout:  float = 0.3
    rsti_dropout: float = 0.1
    head_dropout: float = 0.3
    init_scale: float = 0.1


class CORTEX(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        R, M = cfg.R, cfg.M

        self.z_init = nn.Parameter(torch.randn(R)     * cfg.init_scale)
        self.U_init = nn.Parameter(torch.randn(R, M)  * cfg.init_scale)

        self.attn = IdentityAwareAttention(
            R=R, d_attn=cfg.d_attn,
            position_mode=cfg.position_mode,
            attn_kind=cfg.attn_kind,
            nerf_n_freqs=cfg.nerf_n_freqs,
            include_bold_in_feat=cfg.include_bold_in_feat,
        )
        self.hebbian = None
        if cfg.attn_kind == 'hebbian':
            self.hebbian = HebbianModule(
                R=R,
                alpha_init=cfg.hebbian_alpha_init,
                rate_init_std=cfg.hebbian_rate_init_std,
                H_clip=cfg.hebbian_H_clip,
            )

        self.prp = PerRegionProjection(R=R, d_hidden=cfg.prp_d_hidden,
                                          dropout=cfg.prp_dropout)
        self.rsr = RSTI(
            R=R, M_max=M, d_hidden=cfg.d_hidden_rsr, dropout=cfg.rsti_dropout,
            sharpness=cfg.rsti_sharpness,
            init_logits_std=cfg.rsti_init_logits_std,
            diversity_weight=cfg.rsti_diversity_weight,
        )

        if cfg.head == 'brainnet':
            self.head = BrainNetCNNHead(
                R=R, n_classes=cfg.n_classes,
                e2e_ch=cfg.cls_e2e_filters,
                e2n_ch=cfg.cls_e2n_filters,
                fc_hidden=cfg.cls_fc_hidden,
                dropout=cfg.head_dropout,
            )
        elif cfg.head == 'mlp':
            self.head = MLPHead(
                R=R, n_classes=cfg.n_classes,
                hidden=tuple(cfg.mlp_head_hidden),
                dropout=cfg.head_dropout,
                use_upper_tri=cfg.mlp_head_upper_tri,
                activation=cfg.mlp_head_activation,
            )
        else:
            raise ValueError('unknown head: ' + repr(cfg.head))

    def set_centroids(self, centroids):
        if self.cfg.position_mode in ('nerf', 'nerf_plus_onehot'):
            self.attn.set_centroids(centroids)

    @staticmethod
    def znorm_inner(Z, eps=1e-6):
        t = Z.shape[-1]
        if t < 2:
            return torch.einsum('brt, bst -> brs', Z, Z)
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        Zn = Zc / (Zc.std(dim=-1, keepdim=True) + eps)
        return torch.einsum('brt, bst -> brs', Zn, Zn) / t

    def forward(self, bold):
        B, R, T = bold.shape

        z   = self.z_init.unsqueeze(0).expand(B, -1).contiguous()
        U_t = self.U_init.unsqueeze(0).expand(B, -1, -1).contiguous()
        Z_list = [z]

        H_t = None
        if self.hebbian is not None:
            H_t = torch.zeros(B, R, R, device=bold.device, dtype=bold.dtype)

        for t in range(T):
            S_t = self.znorm_inner(torch.stack(Z_list, dim=-1))

            hebbian_bias = None
            if self.hebbian is not None:
                H_t, hebbian_bias = self.hebbian.update(S_t, H_t)

            x_tilde_t, _, _ = self.attn(S_t, bold[:, :, t], hebbian_bias)
            u_t = self.prp(z, x_tilde_t)
            U_t = torch.cat([U_t[:, :, 1:], u_t.unsqueeze(-1)], dim=-1)
            z   = self.rsr(U_t)
            Z_list.append(z)

        Z_stack = torch.stack(Z_list, dim=-1)
        S_final = self.znorm_inner(Z_stack)
        logits  = self.head(S_final)

        return {'logits': logits, 'z_trajectory': Z_stack, 'S_final': S_final}

    def diversity_loss(self):
        if hasattr(self.rsr, 'diversity_loss'):
            return self.rsr.diversity_loss()
        return None


def build_cortex(cfg, centroids=None):
    model = CORTEX(cfg)
    if cfg.position_mode in ('nerf', 'nerf_plus_onehot'):
        assert centroids is not None
        model.set_centroids(centroids)
    return model
