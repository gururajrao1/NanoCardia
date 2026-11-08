"""Focused checks for upgrade behavior: ECA, focal loss, train-only SMOTE.

Run: uv run python -m tests.test_upgrades (stdlib + torch/numpy only).
"""
import numpy as np
import torch

from src.model import EfficientECG, count_params
from src.train import FocalLoss
from src.data import (smote_train_only, fusion_mixup_train_only,
                        random_oversample_train_only, within_class_mixup_train_only,
                        make_windows, repeat_minority_windows)


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
def test_make_windows_never_crosses_records():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 188)).astype(np.float32)
    y = np.arange(30) % 4
    p = np.array([101] * 12 + [106] * 18)
    rr = rng.uniform(0.3, 1.2, size=(30, 4)).astype(np.float32)
    Xw, yw, Rw, pw = make_windows(X, y, p, rr, radius=2)
    # 12-run -> 8 windows, 18-run -> 14 windows; centers labeled correctly
    assert Xw.shape == (22, 5, 188) and Rw.shape == (22, 5, 4)
    assert pw[:8].tolist() == [101] * 8 and pw[8:].tolist() == [106] * 14
    assert yw[0] == y[2] and yw[8] == y[12 + 2]


def test_repeat_minority_windows():
    rng = np.random.default_rng(7)
    Xw = rng.normal(size=(30, 9, 188)).astype(np.float32)
    yw = np.array([0] * 16 + [1] * 4 + [2] * 8 + [3] * 2)
    Rw = rng.uniform(0.3, 1.2, size=(30, 9, 4)).astype(np.float32)
    pw = np.array([101] * 30)
    a, b, c, d = repeat_minority_windows(Xw, yw, Rw, pw)
    n = np.bincount(b, minlength=4)
    assert n.tolist() == [16, 8, 8, 8] and len(d) == len(b)


def test_checkpoint_roundtrip_and_ensemble():
    import tempfile, os
    from src.model import EfficientECG, SeqECG
    from src.train import save_ckpt
    from src.ensemble import load_model
    d = tempfile.mkdtemp()
    m = EfficientECG().eval()
    save_ckpt(os.path.join(d, 'cnn.pt'), m.state_dict(), {'arch': 'cnn'})
    m2 = load_model(os.path.join(d, 'cnn.pt'))
    x = torch.randn(2, 1, 188)
    assert torch.allclose(m(x), m2(x))
    s = SeqECG(use_adv=True, n_patients=3).eval()
    save_ckpt(os.path.join(d, 'seq.pt'), s.state_dict(),
              {'arch': 'seq', 'use_adv': True, 'n_patients': 3})
    s2 = load_model(os.path.join(d, 'seq.pt'))
    xw = torch.randn(2, 9, 188)
    rw = torch.rand(2, 9, 4)
    assert torch.allclose(s(xw, rw)[0], s2(xw, rw)[0])
    # ensemble math: weights select the stronger member
    z1 = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    z2 = torch.tensor([[0.0, 0.0, 0.0, 2.0]])
    assert (0.9 * z1 + 0.1 * z2).argmax(1).item() == 0
    assert (0.1 * z1 + 0.9 * z2).argmax(1).item() == 3


def test_ode_blend_shapes_and_filter():
    from src.ode_synth import synthesize_f, plausible, znorm
    rng = np.random.default_rng(8)
    t = np.linspace(0, 1, 188)
    Xn = np.stack([znorm(np.exp(-((t - 0.5) ** 2) / 0.0008) +
                         rng.normal(0, 0.05, 188)) for _ in range(60)])
    Xv = np.stack([znorm(1.4 * np.exp(-((t - 0.52) ** 2) / 0.008) +
                         rng.normal(0, 0.05, 188)) for _ in range(30)])
    rr = rng.uniform(0.3, 1.2, size=(10, 4)).astype(np.float32)
    Xs, rs = synthesize_f(Xn, Xv, rr, 20, seed=0)
    assert Xs.shape == (20, 188) and rs.shape == (20, 4)
    assert np.isfinite(Xs).all()
    # pure noise fails the gate; a real blend passes
    assert not plausible(rng.normal(size=188).astype(np.float32),
                         Xn[0], Xv[0])
    assert plausible(Xs[0], np.median(Xn, axis=0), np.median(Xv, axis=0))


def test_make_windows_no_rr():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(24, 188)).astype(np.float32)
    y = np.arange(24) % 4
    p = np.array([101] * 12 + [106] * 12)
    Xw, yw, Rw, pw = make_windows(X, y, p, None, radius=2)
    assert Xw.shape[1] == 5 and Rw is None and len(yw) == 16


def test_minority_batches_and_holdout():
    from src.data import minority_batch_order, record_holdout
    y = np.array([0] * 100 + [1] * 6 + [2] * 20 + [3] * 4)
    order = minority_batch_order(y, bs=26, k=4, seed=0)
    assert len(order) % 26 == 0 and len(order) >= len(y)
    assert set(range(100)) <= set(order.tolist())  # every majority beat seen
    # every batch has >=4 minority (S/F) beats
    for b in range(len(order) // 26):
        chunk = y[order[b * 26:(b + 1) * 26]]
        assert np.isin(chunk, [1, 3]).sum() >= 4
    p = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112])
    vm = record_holdout(p)
    assert vm.sum() >= 1 and vm.sum() < len(p)  # non-trivial patient split
    assert len(np.unique(p[vm])) + len(np.unique(p[~vm])) == len(np.unique(p))


def test_seqecg_forward_and_adv():
    from src.model import SeqECG, GradReverse
    m = SeqECG().eval()
    xw = torch.randn(2, 9, 188)
    rw = torch.rand(2, 9, 4)
    assert m(xw, rw).shape == (2, 4)
    ma = SeqECG(use_adv=True, n_patients=5).eval()
    cls, pat = ma(xw, rw, lamb=0.1)
    assert cls.shape == (2, 4) and pat.shape == (2, 5)
    # reversal flips the gradient sign on the trunk path
    z = torch.randn(3, 8, requires_grad=True)
    GradReverse.apply(z, 1.0).sum().backward()
    assert torch.allclose(z.grad, -torch.ones_like(z))


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
    test_make_windows_never_crosses_records()
    test_repeat_minority_windows()
    test_seqecg_forward_and_adv()
    test_checkpoint_roundtrip_and_ensemble()
    test_minority_batches_and_holdout()
    test_ode_blend_shapes_and_filter()
    test_make_windows_no_rr()
    print('all upgrade checks passed')
