"""Closed-form continuous-time (CfC) rhythm head (Hasani et al. 2022).

Replaces the BiLSTM tail of SeqECG with a tiny liquid layer. The update
is the closed form of dx/dt = -[1/tau + f(x,I)]*x + f(x,I)*A:

    x' = sigmoid(-f) * x + (1 - sigmoid(-f)) * h

f (input-adaptive time constant) and h (candidate) are small dense maps
on [state; trunk embedding; dt]. dt is the beat's normalized pre-RR, so
rhythm timing explicitly paces the recurrence. O(T*d), no solver, CPU
and int8 friendly. Same 160-dim trunk as SeqECG; head is ~6k params vs
~22k for the BiLSTM tail.
"""
import torch
import torch.nn as nn


class CfCLayer(nn.Module):
    """One liquid layer: state dim d, input dim (feat + 1 dt)."""

    def __init__(self, feat_dim, d=16):
        super().__init__()
        self.d = d
        inp = d + feat_dim + 1  # [x; e; dt]
        self.f = nn.Linear(inp, d)
        self.h = nn.Linear(inp, d)

    def forward(self, e, dt):
        """e: B,T,F trunk embeddings; dt: B,T normalized pre-RR."""
        b, t, _ = e.shape
        x = torch.zeros(b, self.d, device=e.device, dtype=e.dtype)
        for i in range(t):
            u = torch.cat([x, e[:, i], dt[:, i:i + 1]], dim=1)
            gate = torch.sigmoid(-self.f(u))          # input-adaptive tau
            cand = torch.tanh(self.h(u))
            x = gate * x + (1 - gate) * cand
        return x


class CfCECG(nn.Module):
    """Window classifier: shared EfficientECG trunk + CfC head, center labeled."""

    def __init__(self, trunk=None, n_classes=4, hidden=16,
                 use_adv=False, n_patients=0, feat_dim=160):
        super().__init__()
        from src.model import EfficientECG
        self.trunk = trunk or EfficientECG()
        self.cfc = CfCLayer(feat_dim, hidden)
        self.fc = nn.Linear(hidden, n_classes)
        self.use_adv = use_adv
        self.pat = nn.Linear(hidden, n_patients) if use_adv else None

    def forward(self, xw, rw=None, lamb=0.0):
        b, t, L = xw.shape
        f = self.trunk.features(xw.reshape(b * t, 1, L),
                                rw.reshape(b * t, -1) if rw is not None else None)
        if rw is not None:
            # per-step pacing from each beat's own normalized pre-RR
            dt = rw[..., 0].clamp(-3, 3)
        else:
            dt = torch.zeros(b, t, device=xw.device, dtype=xw.dtype)
        c = self.cfc(f.view(b, t, -1), dt)
        if self.use_adv:
            from src.model import GradReverse
            return self.fc(c), self.pat(GradReverse.apply(c, lamb))
        return self.fc(c)
