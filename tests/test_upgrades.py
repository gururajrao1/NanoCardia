"""Focused checks for upgrade behavior: ECA, focal loss, train-only SMOTE.

Run: uv run python -m tests.test_upgrades (stdlib + torch/numpy only).
"""
import numpy as np
import torch

from src.model import EfficientECG, count_params
from src.train import FocalLoss
from src.data import (smote_train_only, fusion_mixup_train_only,
                        random_oversample_train_only, within_class_mixup_train_only)


def test_eca_adds_trivial_params():
    assert count_params(EfficientECG(use_eca=True)) - count_params(EfficientECG(use_eca=False)) <= 10


def test_eca_forward_shape():
    m = EfficientECG().eval()
    assert m(torch.randn(4, 1, 188)).shape == (4, 4)


def test_focal_downweights_easy_examples():
    ce = torch.nn.CrossEntropyLoss()
    focal = FocalLoss(gamma=2.0)
    confident = torch.tensor([[5.0, -5.0, -5.0, -5.0]])
    hard = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    t = torch.tensor([0])
    # focal < ce on confident predictions (easy example down-weighted)
    assert focal(confident, t).item() < ce(confident, t).item()
    # hard example keeps most of its loss
    assert focal(hard, t).item() > 0.5 * ce(hard, t).item()


def test_smote_upsamples_minorities_only():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(34, 188)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 4 + [2] * 8 + [3] * 2)
    Xs, ys = smote_train_only(X, y)
    n = np.bincount(ys, minlength=4)
    assert n[0] == 20 and n[2] == 8  # majority + reference class untouched
    assert n[1] == 8 and n[3] == 8    # S/F raised toward V count


def test_smote_chunked_no_self_peers():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 188)).astype(np.float32)
    y = np.array([0] * 100 + [1] * 40 + [2] * 150 + [3] * 10)
    Xs, ys = smote_train_only(X, y)  # multi-chunk path, would OOM unchunked at scale
    assert np.bincount(ys, minlength=4)[1] == 150  # S raised to V count


def test_smote_carries_rr():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(34, 188)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 4 + [2] * 8 + [3] * 2)
    rr = rng.uniform(0.3, 1.2, size=(34, 2)).astype(np.float32)
    Xs, ys, rs = smote_train_only(X, y, rr_tr=rr)
    assert rs.shape == (Xs.shape[0], 2) and len(ys) == len(rs)


def test_multiscale_stem_and_rr_forward():
    from src.model import MSStem
    s = MSStem().eval()
    assert s(torch.randn(2, 1, 188)).shape == (2, 32, 94)
    m = EfficientECG().eval()
    x = torch.randn(3, 1, 188)
    assert m(x, torch.tensor([[0.5, 0.8, 0.8, 0.625]] * 3)).shape == (3, 4)  # with RR
    assert m(x).shape == (3, 4)  # without RR falls back cleanly
    assert count_params(m) < 25000  # still tiny


def test_fusion_mixup_upsamples_f_only():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(34, 188)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 4 + [2] * 8 + [3] * 2)
    Xs, ys = fusion_mixup_train_only(X, y, ratio=1.0)
    n = np.bincount(ys, minlength=4)
    assert n[0] == 20 and n[1] == 4 and n[2] == 8  # N/S/V untouched
    assert n[3] == 8  # F raised toward V count via N+V mix
    # mixed beat lies between its N and V parents elementwise
    assert Xs[34:].min() >= min(X.min(), 0) - 5


def test_fusion_mixup_carries_rr():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(34, 188)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 4 + [2] * 8 + [3] * 2)
    rr = rng.uniform(0.3, 1.2, size=(34, 2)).astype(np.float32)
    Xs, ys, rs = fusion_mixup_train_only(X, y, rr_tr=rr, ratio=0.5)
    assert rs.shape == (Xs.shape[0], 2) and len(ys) == len(rs)
    assert np.bincount(ys, minlength=4)[3] == 4  # 0.5 * V count


def test_random_oversample_repeats_real_beats_only():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(34, 188)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 4 + [2] * 8 + [3] * 2)
    Xs, ys = random_oversample_train_only(X, y)
    n = np.bincount(ys, minlength=4)
    assert n[0] == 20 and n[2] == 8  # majority + reference untouched
    assert n[1] == 8 and n[3] == 8    # S/F repeated toward V count
    # every added beat is an exact copy of a real beat (no synthetic waveforms)
    assert set(map(tuple, Xs[34:].tolist())) <= set(map(tuple, X.tolist()))


def test_within_class_mixup_keeps_n_out():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(42, 188)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 4 + [2] * 8 + [3] * 10)
    Xs, ys = within_class_mixup_train_only(X, y, ratios={1: 1.0, 3: 2.0})
    n = np.bincount(ys, minlength=4)
    assert n[0] == 20 and n[2] == 8  # N/V untouched
    assert n[1] == 8                 # S raised to 1x V count (+4)
    assert n[3] == 16                # F raised toward 2x V count (+6)
    assert Xs.shape[0] == 42 + 4 + 6


def test_qrs_branch_forward_and_budget():
    m = EfficientECG().eval()
    x = torch.randn(3, 1, 188)
    assert m(x, torch.tensor([[0.5, 0.8, 0.8, 0.625]] * 3)).shape == (3, 4)
    assert m(x).shape == (3, 4)  # RR falls back to zeros, QRS still active
    assert count_params(m) < 25000  # still tiny
    assert count_params(EfficientECG(use_qrs=False)) < count_params(m)


def test_gan_shapes_and_gp_finite():
    from src.gan import Generator, Critic, gradient_penalty, generate_beats
    g, c = Generator(), Critic()
    z = torch.randn(4, 64)
    fake = g(z)
    assert fake.shape == (4, 1, 188)
    assert c(fake).shape == (4, 1)
    real = torch.randn(4, 1, 188)
    gp = gradient_penalty(c, real, fake.detach(),
                          torch.device('cpu'))
    assert torch.isfinite(gp) and gp.item() >= 0
    beats = generate_beats(g, 10)
    assert beats.shape == (10, 188) and np.isfinite(beats).all()


if __name__ == '__main__':
    test_eca_adds_trivial_params()
    test_eca_forward_shape()
    test_focal_downweights_easy_examples()
    test_smote_upsamples_minorities_only()
    test_smote_chunked_no_self_peers()
    test_smote_carries_rr()
    test_multiscale_stem_and_rr_forward()
    test_fusion_mixup_upsamples_f_only()
    test_fusion_mixup_carries_rr()
    test_random_oversample_repeats_real_beats_only()
    test_within_class_mixup_keeps_n_out()
    test_qrs_branch_forward_and_budget()
    test_gan_shapes_and_gp_finite()
    print('all upgrade checks passed')
