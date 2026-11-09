"""Matched-filter stem init (Farag/Saeed Sensors 2023 prior).

Each conv1 filter starts as a DS1 class-median beat (N/S/V/F templates,
train side only, never test). A convolution layer in deep-learning
convention computes cross-correlation, so initializing weights with the
template itself makes each filter a correlation detector for its class:
y[n] = sum_m x[n+m] * t[m]. (DSP texts say "time-reversed" because they
assume convolution; here no reversal is needed.)

F template is shrunk toward V (few DS1 F beats -> noisy median).
Freeze-then-finetune: stem grads are zeroed for the first `freeze`
epochs, then released (ablate freeze 0 vs 5).
"""
import numpy as np
import torch


def class_templates(X, y, n_classes=4, shrink_f=0.5):
    """Per-class median beats from train data only. F shrunk toward V."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    temps = []
    for c in range(n_classes):
        pool = X[y == c]
        if len(pool) == 0:
            pool = X
        temps.append(np.median(pool, axis=0).astype(np.float32))
    temps[3] = ((1.0 - shrink_f) * temps[3] + shrink_f * temps[2]).astype(np.float32)
    return temps


def _prep_kernel(template, k):
    """Resample template to kernel length k and z-norm."""
    t = torch.from_numpy(np.asarray(template, dtype=np.float32))
    t = torch.nn.functional.interpolate(t.view(1, 1, -1), size=k,
                                        mode='linear', align_corners=False).view(-1)
    t = t - t.mean()
    s = t.std()
    return (t / s) if s > 1e-6 else t


def _fill_conv(conv, templates):
    """Tile 4 class templates across conv's out-channels (4 copies each)."""
    k = conv.weight.shape[2]
    kernels = [_prep_kernel(t, k) for t in templates]
    with torch.no_grad():
        for i in range(conv.weight.shape[0]):
            conv.weight[i, 0].copy_(kernels[i % len(kernels)])
            if conv.bias is not None:
                conv.bias[i].zero_()


def init_stem_matched(model, templates):
    """In-place matched init of MSStem branches (SelfONN-safe).

    Plain conv branches get templates directly. SelfONN branches get
    templates on the q=1 conv and zeros on q>1 convs, so the stem starts
    exactly linear (matched) and learns nonlinearity from there.
    """
    from src.model import SelfONN1d
    for branch in (model.stem.b7[0], model.stem.b15[0]):
        if isinstance(branch, SelfONN1d):
            _fill_conv(branch.convs[0], templates)
            with torch.no_grad():
                for higher in branch.convs[1:]:
                    higher.weight.zero_()
        else:
            _fill_conv(branch, templates)
    return model


def stem_conv_params(model):
    """Iterate stem conv weight params (for freeze/unfreeze)."""
    from src.model import SelfONN1d
    for branch in (model.stem.b7[0], model.stem.b15[0]):
        if isinstance(branch, SelfONN1d):
            for c in branch.convs:
                yield c.weight
        else:
            yield branch.weight


def zero_stem_grads(model):
    """Zero stem conv grads (freeze-then-finetune phase)."""
    for p in stem_conv_params(model):
        if p.grad is not None:
            p.grad.detach().zero_()
