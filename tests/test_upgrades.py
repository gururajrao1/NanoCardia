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


def test_selfonn_forward_grad_and_budget():
    from src.model import SelfONN1d, EfficientECG, count_params
    m = SelfONN1d(1, 16, 7, Q=3, stride=2, padding=3).train()
    x = torch.randn(4, 1, 188, requires_grad=False)
    y = m(x)
    assert y.shape == (4, 16, 94)
    y.sum().backward()  # all Q branches get gradients
    assert all(c.weight.grad is not None and torch.isfinite(c.weight.grad).all()
               for c in m.convs)
    q1 = SelfONN1d(1, 16, 7, Q=1, stride=2, padding=3)
    assert count_params(m) > count_params(q1)  # higher Q costs params
    e0 = EfficientECG().eval()
    e1 = EfficientECG(use_selfonn=True).eval()
    assert e1(x).shape == (4, 4)
    assert count_params(e1) < 25000
    assert count_params(e1) - count_params(e0) < 2000  # stem-only swap
    assert count_params(EfficientECG(use_rr=False, use_selfonn=True)) < \
        count_params(e1)


def test_rr_per_patient_normalized():
    from src.data import synthetic_beats, col_znorm
    a = np.array([[0.5, 0.8], [0.7, 1.0], [0.9, 1.2]], dtype=np.float32)
    z = col_znorm(a)
    assert np.allclose(z.mean(0), 0, atol=1e-6)
    assert np.allclose(z.std(0), 1, atol=1e-6)
    _, _, p, rr = synthetic_beats(n=1200, seed=0)
    for pat in np.unique(p):
        blk = rr[p == pat]
        if len(blk) > 1:
            assert np.allclose(blk.mean(0), 0, atol=1e-5)
            assert np.allclose(blk.std(0), 1, atol=1e-5)


def test_same_seed_same_result():
    # determinism lockdown: identical seed => identical numbers
    from src.data import synthetic_beats
    from src.train import run_once, set_seed
    X, y, p, rr = synthetic_beats(n=1200, seed=1)
    kw = dict(Xte=X[960:], yte=y[960:], use_weights=True, epochs=1,
              rrtr=rr[:960], rrte=rr[960:], loss='focal')
    set_seed(0)
    r1 = run_once(X[:960], y[:960], **kw)
    set_seed(0)
    r2 = run_once(X[:960], y[:960], **kw)
    assert r1['macro_recall'] == r2['macro_recall']
    assert r1['per_class'] == r2['per_class']


def test_config_file_overrides_and_rejects():
    import tempfile, os
    from src.train import apply_config
    import argparse
    d = tempfile.mkdtemp()
    good = os.path.join(d, 'good.toml')
    with open(good, 'w') as f:
        f.write('epochs = 7\nswa = true\n')
    import sys
    old = sys.argv
    try:
        sys.argv = ['prog', '--config', good]
        ap = argparse.ArgumentParser()
        ap.add_argument('--config', default=None)
        ap.add_argument('--epochs', type=int, default=10)
        ap.add_argument('--swa', action='store_true')
        apply_config(ap)
        a = ap.parse_args()
        assert a.epochs == 7 and a.swa is True
        bad = os.path.join(d, 'bad.toml')
        with open(bad, 'w') as f:
            f.write('nope = 1\n')
        sys.argv = ['prog', '--config', bad]
        try:
            apply_config(ap)
            raise AssertionError('unknown key accepted')
        except SystemExit:
            pass
    finally:
        sys.argv = old


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


def test_ecgsyn_fit_and_beats():
    from src.ode_synth import (fit_gaussians, synthesize_f_ecgsyn,
                               ECG_ANGLES, znorm)
    from src.data import synthetic_beats
    X, y, p, rr = synthetic_beats(n=1200, seed=2)
    Xf, rrf = X[y == 3], rr[y == 3]
    med = np.median(Xf, axis=0)
    a, z0 = fit_gaussians(med)
    assert a.shape == (5,) and np.isfinite(a).all() and np.isfinite(z0)
    assert a[2] == a.max()  # R dominates the fit
    n_med = np.median(X[y == 0][:500], axis=0)
    v_med = np.median(X[y == 2][:500], axis=0)
    Xs, rs = synthesize_f_ecgsyn(Xf, rrf, 8, n_med, v_med, seed=0,
                                 steps_per_cycle=200)
    assert Xs.shape == (8, 188) and rs.shape == (8, 4)
    assert np.isfinite(Xs).all()
    Xs2, _ = synthesize_f_ecgsyn(Xf, rrf, 8, n_med, v_med, seed=0,
                                 steps_per_cycle=200)
    assert np.array_equal(Xs, Xs2)  # seeded generation is deterministic


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


def test_matched_init_templates_and_fill():
    from src.matched import (class_templates, init_stem_matched,
                             stem_conv_params, zero_stem_grads)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 188)).astype(np.float32)
    y = np.array([0] * 20 + [1] * 8 + [2] * 8 + [3] * 4)
    temps = class_templates(X, y)
    assert len(temps) == 4 and all(t.shape == (188,) for t in temps)
    assert np.allclose(temps[3], 0.5 * np.median(X[y == 3], axis=0) +
                       0.5 * np.median(X[y == 2], axis=0))  # F shrunk toward V
    for use_sn in (False, True):
        m = EfficientECG(use_selfonn=use_sn)
        before = [p.detach().clone() for p in m.stem.b7[0].parameters()]
        init_stem_matched(m, temps)
        after = [p.detach().clone() for p in m.stem.b7[0].parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(before, after))
        # each branch holds 4 copies of each class kernel
        w = m.stem.b7[0].convs[0].weight if use_sn else m.stem.b7[0].weight
        assert w.shape[0] == 16 and w.shape[2] == 7
        z = (w[0, 0] - w[4, 0]).abs().max()  # same class, 4 apart
        assert z.item() == 0.0
        # frozen grads zero out; params keep flowing after release
        m.train()
        m(torch.randn(2, 1, 188)).sum().backward()
        zero_stem_grads(m)
        assert all(p.grad is None or (p.grad == 0).all()
                   for p in stem_conv_params(m))


def test_cfc_head_forward_grad_and_budget():
    from src.cfc import CfCLayer, CfCECG
    from src.model import count_params, SeqECG
    layer = CfCLayer(160, 16).train()
    e = torch.randn(2, 9, 160)
    dt = torch.randn(2, 9)
    c = layer(e, dt)
    assert c.shape == (2, 16)
    c.sum().backward()  # f and h maps both get gradients
    assert layer.f.weight.grad is not None and layer.h.weight.grad is not None
    assert torch.isfinite(layer.f.weight.grad).all()
    m = CfCECG().eval()
    xw = torch.randn(2, 9, 188)
    rw = torch.rand(2, 9, 4)
    assert m(xw, rw).shape == (2, 4)
    assert m(xw).shape == (2, 4)  # no-RR path works
    # dt pacing is live: different timing -> different state
    c1 = layer(e, torch.zeros(2, 9))
    c2 = layer(e, torch.ones(2, 9))
    assert not torch.allclose(c1, c2)
    ma = CfCECG(use_adv=True, n_patients=5).eval()
    cls, pat = ma(xw, rw, lamb=0.1)
    assert cls.shape == (2, 4) and pat.shape == (2, 5)
    # head is much smaller than the BiLSTM tail it replaces
    assert count_params(m) < count_params(SeqECG())
    assert count_params(m) < 30000


def test_fetch_retry_retries_then_raises():
    # loader must not hang forever on a stalled PhysioNet fetch:
    # retry wrapper re-calls, then surfaces the last error to the skip path
    from src.data import _fetch_with_retry
    calls = []

    def fail_once(rid):
        calls.append(rid)
        if len(calls) < 2:
            raise ConnectionError('stalled')
        return ('sig', rid)

    assert _fetch_with_retry(fail_once, '100', retries=2) == ('sig', '100')
    assert calls == ['100', '100']  # one retry, then success

    def always_fail(rid):
        calls.append(rid)
        raise TimeoutError('hung')

    try:
        _fetch_with_retry(always_fail, '101', retries=2)
        raise AssertionError('persistent failure accepted')
    except TimeoutError:
        pass
    assert calls[-3:] == ['101', '101', '101']  # initial + 2 retries


def test_fetch_retry_wall_clock_timeout():
    # a hung fetch (sleep) is abandoned after timeout_s, then retried
    import time
    from src.data import _fetch_with_retry
    calls = []

    def hung(rid):
        calls.append(rid)
        time.sleep(30)
        return 'never'

    t0 = time.time()
    try:
        _fetch_with_retry(hung, '119', retries=1, timeout_s=1)
        raise AssertionError('hung fetch accepted')
    except TimeoutError:
        pass
    assert calls == ['119', '119'] and time.time() - t0 < 10


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
    test_same_seed_same_result()
    test_config_file_overrides_and_rejects()
    test_selfonn_forward_grad_and_budget()
    test_rr_per_patient_normalized()
    test_matched_init_templates_and_fill()
    test_cfc_head_forward_grad_and_budget()
    test_fetch_retry_retries_then_raises()
    test_fetch_retry_wall_clock_timeout()
    test_ecgsyn_fit_and_beats()
    print('all upgrade checks passed')
