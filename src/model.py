"""Efficient depthwise-separable 1D CNN (~45k params) vs vanilla baseline."""
import torch
import torch.nn as nn


class DSBlock(nn.Module):
    def __init__(self, c_in, c_out, k=7, s=2, p=3):
        super().__init__()
        self.depth = nn.Conv1d(c_in, c_in, k, stride=s, padding=p, groups=c_in, bias=False)
        self.point = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.res = (c_in == c_out and s == 1)
        # cheap projection when shapes match handled by identity; else no residual
    def forward(self, x):
        y = self.act(self.bn(self.point(self.depth(x))))
        return (x + y) / 1.414 if self.res and x.shape == y.shape else y


class ECA1d(nn.Module):
    """Efficient channel attention: 1D conv on channel-pooled descriptor.

    ECA replaces SE's two FC layers (2C^2/r params) with one 1D conv
    (k params, k~3-5). Research: +59 params on MobileNetV2, beats SE.
    """
    def __init__(self, c, gamma=2, b=1):
        super().__init__()
        import math
        k = int(abs((math.log2(c) + b) / gamma))
        k = k if k % 2 else k + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        y = x.mean(-1, keepdim=True).transpose(1, 2)  # B,1,C
        y = self.conv(y).transpose(1, 2)              # B,C,1
        return x * torch.sigmoid(y)


class MSStem(nn.Module):
    """Multi-scale stem: parallel k=7 + k=15 convs, concat to w channels.

    KAN-Former ablation: removing the multi-scale block cost 12.9% accuracy.
    Two short kernels cover narrow (N/S) and wide (V/F) QRS in one pass.
    """
    def __init__(self, w=32):
        super().__init__()
        w1, w2 = w // 2, w - w // 2
        self.b7 = nn.Sequential(
            nn.Conv1d(1, w1, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(w1), nn.ReLU(inplace=True))
        self.b15 = nn.Sequential(
            nn.Conv1d(1, w2, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(w2), nn.ReLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.b7(x), self.b15(x)], dim=1)


class EfficientECG(nn.Module):
    """~18k params: multi-scale stem + 3 DS blocks + ECA + QRS branch + RR + 1 FC."""
    def __init__(self, n_classes=4, w=32, use_eca=True, use_rr=True, use_qrs=True):
        super().__init__()
        self.stem = MSStem(w)
        self.b1 = DSBlock(w, w * 2)       # 94 -> 47
        self.b2 = DSBlock(w * 2, w * 2, s=1)  # residual block
        self.b3 = DSBlock(w * 2, w * 4)   # 47 -> 24
        self.eca = ECA1d(w * 4) if use_eca else nn.Identity()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.use_rr = use_rr
        self.rr_mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU(inplace=True),
                                   nn.Linear(16, 16), nn.ReLU(inplace=True))
        self.use_qrs = use_qrs
        # QRS-local branch: fusion is decided in local QRS detail (Shi 3-branch
        # topology). Center 64-sample crop, narrow kernel, GAP to 16-dim.
        self.qrs = nn.Sequential(nn.Conv1d(1, 16, 7, stride=2, padding=3, bias=False),
                                 nn.BatchNorm1d(16), nn.ReLU(inplace=True))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(w * 4 + (16 if use_rr else 0) + (16 if use_qrs else 0),
                            n_classes)

    def features(self, x, rr=None):
        """160-dim per-beat embedding (GAP morphology + RR MLP + QRS)."""
        q = None
        if self.use_qrs:
            c = x.size(-1) // 2
            q = self.gap(self.qrs(x[..., c - 32:c + 32])).squeeze(-1)
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.eca(x)
        x = self.gap(x).squeeze(-1)
        if self.use_rr:
            if rr is None:
                rr = torch.zeros(x.size(0), 4, device=x.device, dtype=x.dtype)
            x = torch.cat([x, self.rr_mlp(rr)], dim=1)
        if self.use_qrs:
            x = torch.cat([x, q], dim=1)
        return x

    def forward(self, x, rr=None):
        return self.fc(self.drop(self.features(x, rr)))


class GradReverse(torch.autograd.Function):
    """Gradient reversal for domain-adversarial training (Ganin et al.)."""
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lamb * grad_out, None


class SeqECG(nn.Module):
    """Beat-sequence classifier: shared trunk embeds each beat of a window,
    a tiny BiLSTM reads rhythm context, center beat is classified.

    Isolated-beat models cap out on F (morphologically ambiguous alone but
    usually obvious in rhythm context). Windows never cross patients.
    """
    def __init__(self, trunk=None, n_classes=4, hidden=16,
                 use_adv=False, n_patients=0):
        super().__init__()
        self.trunk = trunk or EfficientECG()
        self.lstm = nn.LSTM(160, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, n_classes)
        self.use_adv = use_adv
        # patient head: predicts record id from rhythm features through a
        # gradient-reversal layer, forcing patient-independent representations
        self.pat = nn.Linear(hidden * 2, n_patients) if use_adv else None

    def forward(self, xw, rw=None, lamb=0.0):
        b, t, L = xw.shape
        f = self.trunk.features(xw.reshape(b * t, 1, L),
                                rw.reshape(b * t, -1) if rw is not None else None)
        h, _ = self.lstm(f.view(b, t, -1))
        c = h[:, t // 2]
        if self.use_adv:
            return self.fc(c), self.pat(GradReverse.apply(c, lamb))
        return self.fc(c)


class VanillaCNN(nn.Module):
    """Baseline: 2 standard convs + big FC (~180k params)."""
    def __init__(self, n_classes=4, length=188):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * (length // 4), 128),
                                nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.fc(self.f(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
