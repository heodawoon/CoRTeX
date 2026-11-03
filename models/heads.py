import torch
import torch.nn as nn
import torch.nn.functional as F


class _E2E(nn.Module):
    def __init__(self, in_ch, out_ch, R):
        super().__init__()
        self.row = nn.Conv2d(in_ch, out_ch, kernel_size=(1, R), padding=0)
        self.col = nn.Conv2d(in_ch, out_ch, kernel_size=(R, 1), padding=0)
        self.R   = R

    def forward(self, x):
        r = self.row(x).expand(-1, -1, self.R, -1)
        c = self.col(x).expand(-1, -1, -1, self.R)
        return r + c


class BrainNetCNNHead(nn.Module):
    """
    Kawahara, Jeremy, et al. "BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment." NeuroImage 146 (2017): 1038-1049.
    """
    def __init__(self, R, n_classes, e2e_ch=8, e2n_ch=16, fc_hidden=64, dropout=0.3):
        super().__init__()
        self.R = R
        self.e2e  = _E2E(1, e2e_ch, R)
        self.e2n  = nn.Conv2d(e2e_ch, e2n_ch, kernel_size=(1, R))
        self.fc1  = nn.Linear(e2n_ch * R, fc_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2  = nn.Linear(fc_hidden, n_classes)

    def forward(self, S):
        x = S.unsqueeze(1)
        x = F.leaky_relu(self.e2e(x), 0.1)
        x = F.leaky_relu(self.e2n(x), 0.1)
        x = x.flatten(1)
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = self.drop(x)
        return self.fc2(x)


class MLPHead(nn.Module):
    def __init__(self, R, n_classes, hidden=(1024, 256), dropout=0.3,
                 use_upper_tri=True, activation='relu'):
        super().__init__()
        self.R = R
        self.use_upper_tri = use_upper_tri
        if use_upper_tri:
            iu, ju = torch.triu_indices(R, R, offset=1)
            self.register_buffer('iu', iu)
            self.register_buffer('ju', ju)
            in_dim = iu.numel()
        else:
            in_dim = R * R

        if isinstance(hidden, int):
            hidden = (hidden,)
        act_cls = {'relu': nn.ReLU, 'gelu': nn.GELU,
                   'leaky_relu': lambda: nn.LeakyReLU(0.1)}[activation]

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), act_cls(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, S):
        if self.use_upper_tri:
            x = S[:, self.iu, self.ju]
        else:
            x = S.flatten(1)
        return self.net(x)
