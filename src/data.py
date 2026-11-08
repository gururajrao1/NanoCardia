"""Efficient ECG beat datasets: real MIT-BIH via wfdb + synthetic fallback."""
import numpy as np

AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
}
CLASS_NAMES = ['N', 'S', 'V', 'F']
DE_CHAZAL_DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
                 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DE_CHAZAL_DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
                 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
BEAT_LEN = 188
FS_TARGET = 125


def z_norm(x):
    m, s = x.mean(), x.std() + 1e-8
    return ((x - m) / s).astype(np.float32)


def extract_beats_wfdb(record_ids=None, win=188):
    """Download+cut MIT-BIH beats.

    Returns (X (n,win), y (n,), patients (n,), rr (n,4)) where rr holds
    pre/post R-R, local-mean RR (+-5 beats) and pre/local ratio in seconds.
    Premature (short pre-RR) beats are the clinical signature of S; fusion
    rides near-normal rhythm vs premature V, so timing separates what
    morphology cannot. Per-record median fallback at edges.
    """
    import wfdb
    if record_ids is None:
        record_ids = [f"{i:03d}" for i in
                      [100, 101, 103, 105, 106, 107, 108, 109, 111, 112, 113, 114,
                       115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202,
                       203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 219, 220,
                       221, 222, 223, 228, 230, 231, 232, 233, 234]]
    half = win // 2
    Xs, ys, ps, rs = [], [], [], []
    import sys
    for i, rid in enumerate(record_ids):
        print(f"[{i+1}/{len(record_ids)}] loading {rid}...", flush=True)
        try:
            sig, meta = wfdb.rdsamp(rid, channels=[0], pn_dir='mitdb')
            ann = wfdb.rdann(rid, 'atr', pn_dir='mitdb')
            print(f"  {rid}: {len(ann.sample)} ann, {len(sig)} samples", flush=True)
        except Exception as e:
            print(f"skip {rid}: {e}", file=sys.stderr, flush=True)
            continue
        fs = float(meta['fs'])
        s = sig[:, 0].astype(np.float64)
        # wfdb 360Hz -> downsample to ~125Hz by taking every ~2.88 samples
        # cheap: resample via linear interp to fixed factor
        factor = fs / FS_TARGET
        n_target = int(len(s) / factor)
        xp = np.linspace(0, len(s) - 1, n_target)
        s125 = np.interp(xp, np.arange(len(s)), s)
        samp = np.asarray(ann.sample)
        syms = list(ann.symbol)
        diffs = np.diff(samp) / fs
        med = float(np.median(diffs)) if len(diffs) else 0.8
        for j in range(len(samp)):
            sym = syms[j]
            if sym not in AAMI_MAP:
                continue
            c = int(samp[j] / factor)
            if c - half < 0 or c + half >= len(s125):
                continue
            pre = float(diffs[j - 1]) if j > 0 else med
            post = float(diffs[j]) if j < len(diffs) else med
            lo, hi = max(0, j - 5), min(len(diffs), j + 5)
            local = float(np.mean(diffs[lo:hi])) if hi > lo else med
            local = max(local, 0.2)
            beat = s125[c - half:c + half]
            Xs.append(z_norm(beat))
            ys.append(AAMI_MAP[sym])
            ps.append(int(rid))
            rs.append((pre, post, local, pre / local))
    X = np.stack(Xs).astype(np.float32) if Xs else np.zeros((0, win), np.float32)
    rr = np.array(rs, dtype=np.float32) if rs else np.zeros((0, 4), np.float32)
    return X, np.array(ys, dtype=np.int64), np.array(ps, dtype=np.int64), rr


def _template(cls, t):
    """Hard-but-measurable morphologies: S mimics N, F sits between N and V."""
    if cls == 0:  # N: narrow R
        return np.exp(-((t - 0.5) ** 2) / 0.0008) - 0.15 * np.exp(-((t - 0.38) ** 2) / 0.002)
    if cls == 1:  # S: near-normal narrow R, only slightly premature + smaller
        return 0.92 * np.exp(-((t - 0.485) ** 2) / 0.0009) - 0.12 * np.exp(-((t - 0.375) ** 2) / 0.002)
    if cls == 2:  # V: wide bizarre QRS
        return 1.4 * np.exp(-((t - 0.52) ** 2) / 0.008) - 0.5 * np.exp(-((t - 0.65) ** 2) / 0.004)
    # F: fusion = halfway between N and V, easily confused with both
    return 0.75 * np.exp(-((t - 0.5) ** 2) / 0.002) - 0.25 * np.exp(-((t - 0.60) ** 2) / 0.005)


def smote_train_only(Xtr, ytr, target_classes=(1, 3), seed=0, k=5, rr_tr=None):
    """Minority-only SMOTE on the TRAIN split (test untouched: no leakage).

    Interpolates each minority beat with a random k-NN peer of the same
    class, upsampling S/F toward the V-class count. Research (MAK-Net,
    MIT-BIH): S 92.08%->98.48%, F 90.20%->96.57% with train-only SMOTE.
    Dependency-free: pure numpy.
    """
    rng = np.random.default_rng(seed)
    counts = np.bincount(ytr, minlength=4)
    ref = int(counts[2])  # upsample S/F toward V count
    has_rr = rr_tr is not None
    out_x, out_y = [Xtr], [ytr]
    out_r = [rr_tr] if has_rr else []
    for cls in target_classes:
        m = Xtr[ytr == cls]
        need = max(0, ref - len(m))
        if len(m) < 2 or need == 0:
            continue
        # k-NN peers via chunked pairwise distances (full n*n*dim matrix
        # needs GBs on real data: 2829^2*188 float32 ~6GB)
        n = len(m)
        k_eff = min(k, n - 1)
        nn = np.empty((n, k_eff), dtype=np.int64)
        for s in range(0, n, 128):
            e = min(s + 128, n)
            d2 = ((m[s:e, None, :] - m[None, :, :]) ** 2).sum(-1)
            for bi in range(e - s):
                d2[bi, s + bi] = np.inf  # exclude self
            nn[s:e] = np.argpartition(d2, k_eff - 1, axis=1)[:, :k_eff]
        picks = rng.integers(0, len(m), need)
        peers = nn[picks, rng.integers(0, nn.shape[1], need)]
        gap = rng.uniform(0, 1, need).astype(np.float32)
        synth = (m[picks] + gap[:, None] * (m[peers] - m[picks])).astype(np.float32)
        out_x.append(synth)
        out_y.append(np.full(need, cls, dtype=np.int64))
        if has_rr:
            mr = rr_tr[ytr == cls]
            out_r.append((mr[picks] + gap[:, None] * (mr[peers] - mr[picks])).astype(np.float32))
    if has_rr:
        return np.concatenate(out_x), np.concatenate(out_y), np.concatenate(out_r)
    return np.concatenate(out_x), np.concatenate(out_y)


def fusion_mixup_train_only(Xtr, ytr, seed=0, rr_tr=None, ratio=1.0):
    """Synthesize F (fusion) beats by mixing N + V beats (train only).

    Physiologically motivated: fusion beats ARE simultaneous normal +
    ventricular activation, so N/V interpolation matches their morphology
    better than F/F SMOTE (which overfits 414 narrow examples and blurred
    F into N/V: F 0.39 -> 0.02 last run). RR intervals are mixed the same way.
    Upsamples F toward ratio * V count (default 1.0).
    """
    rng = np.random.default_rng(seed)
    n = np.bincount(ytr, minlength=4)
    need = max(0, int(n[2] * ratio) - n[3])
    xn, xv = Xtr[ytr == 0], Xtr[ytr == 2]
    if len(xn) == 0 or len(xv) == 0 or need == 0:
        if rr_tr is not None:
            return Xtr, ytr, rr_tr
        return Xtr, ytr
    pi_n = rng.integers(0, len(xn), need)
    pi_v = rng.integers(0, len(xv), need)
    lam = rng.uniform(0.3, 0.7, need).astype(np.float32)
    synth = ((1 - lam[:, None]) * xn[pi_n] + lam[:, None] * xv[pi_v]).astype(np.float32)
    out_x = np.concatenate([Xtr, synth])
    out_y = np.concatenate([ytr, np.full(need, 3, dtype=np.int64)])
    if rr_tr is not None:
        rn, rv = rr_tr[ytr == 0], rr_tr[ytr == 2]
        rs = ((1 - lam[:, None]) * rn[pi_n] + lam[:, None] * rv[pi_v]).astype(np.float32)
        return out_x, out_y, np.concatenate([rr_tr, rs])
    return out_x, out_y


def minority_batch_order(y, bs=256, k=4, seed=0):
    """Flat index array where every consecutive bs-chunk holds >=k S and >=k F.

    Per-batch minority guarantee (research #3): smoother gradients than
    dataset-level repetition, no synthetic waveforms. Cycle minority indices.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    mins = np.where(np.isin(y, [1, 3]))[0]
    majs = np.where(~np.isin(y, [1, 3]))[0]
    rng.shuffle(mins)
    rng.shuffle(majs)
    import math
    n_batches = max(math.ceil(len(y) / bs),
                    math.ceil(len(majs) / (bs - k)))
    need_min = n_batches * k
    need_maj = n_batches * (bs - k)
    min_pool = np.tile(mins, max(1, (need_min + len(mins) - 1) // len(mins)))[:need_min]
    maj_pool = np.tile(majs, max(1, (need_maj + len(majs) - 1) // len(majs)))[:need_maj]
    rng.shuffle(min_pool)
    rng.shuffle(maj_pool)
    out = []
    for b in range(n_batches):
        chunk = np.concatenate([min_pool[b * k:(b + 1) * k],
                                maj_pool[b * (bs - k):(b + 1) * (bs - k)]])
        rng.shuffle(chunk)
        out.append(chunk)
    return np.concatenate(out)


def record_holdout(p, every=7):
    """Deterministic patient-disjoint val mask: every 7th sorted record.

    Checkpoint selection on DS1-val, test touched once — the leakage fix
    research #0 demands (best-on-test selection is test peeking).
    """
    p = np.asarray(p)
    recs = np.unique(p)
    val_recs = set(recs[::every].tolist())
    return np.array([r in val_recs for r in p])


def random_oversample_train_only(Xtr, ytr, target_classes=(1, 3), seed=0, rr_tr=None):
    """Repeat real minority beats (train only, no synthetic waveforms).

    Plain random oversampling: the literature (Xing et al. 2020 two-step DNN)
    reports it outperforms SMOTE on MIT-BIH because every training beat stays
    a valid cardiac waveform. Combine with focal loss so repeats don't
    dominate gradients. Test split untouched.
    """
    rng = np.random.default_rng(seed)
    ref = int(np.bincount(ytr, minlength=4)[2])  # toward V count
    out_x, out_y = [Xtr], [ytr]
    out_r = [rr_tr] if rr_tr is not None else []
    for cls in target_classes:
        m = Xtr[ytr == cls]
        need = max(0, ref - len(m))
        if need == 0:
            continue
        picks = rng.integers(0, len(m), need)
        out_x.append(m[picks].astype(np.float32))
        out_y.append(np.full(need, cls, dtype=np.int64))
        if rr_tr is not None:
            out_r.append(rr_tr[ytr == cls][picks])
    if rr_tr is not None:
        return np.concatenate(out_x), np.concatenate(out_y), np.concatenate(out_r)
    return np.concatenate(out_x), np.concatenate(out_y)


def within_class_mixup_train_only(Xtr, ytr, seed=0, rr_tr=None, ratios=None):
    """Beta mixup restricted to same-class S/F pairs, hard labels (train only).

    The ONLY safe waveform augmentation per Zheng et al. (PLOS ONE 2025):
    N stays OUT of the blend pool, labels stay hard, ratio capped ~2x.
    Cross-class N+V blending collapsed real F recall to ~0 in our runs.
    """
    if ratios is None:
        ratios = {1: 1.0, 3: 2.0}
    rng = np.random.default_rng(seed)
    v_count = int(np.bincount(ytr, minlength=4)[2])
    out_x, out_y = [Xtr], [ytr]
    out_r = [rr_tr] if rr_tr is not None else []
    for cls, ratio in ratios.items():
        m = Xtr[ytr == cls]
        need = max(0, int(v_count * ratio) - len(m))
        if len(m) < 2 or need == 0:
            continue
        i1 = rng.integers(0, len(m), need)
        i2 = rng.integers(0, len(m), need)
        lam = rng.beta(0.5, 0.5, need).astype(np.float32)
        out_x.append(((lam[:, None] * m[i1] + (1 - lam[:, None]) * m[i2]).astype(np.float32)))
        out_y.append(np.full(need, cls, dtype=np.int64))
        if rr_tr is not None:
            mr = rr_tr[ytr == cls]
            out_r.append((lam[:, None] * mr[i1] + (1 - lam[:, None]) * mr[i2]).astype(np.float32))
    if rr_tr is not None:
        return np.concatenate(out_x), np.concatenate(out_y), np.concatenate(out_r)
    return np.concatenate(out_x), np.concatenate(out_y)


def make_windows(X, y, p, rr, radius=4):
    """Sliding windows of 2*radius+1 consecutive beats; label = center beat.

    Windows never cross records (rhythm context stays within one patient).
    Beats arrive record-by-record in annotation order, so same-record runs
    are contiguous; grouping is still done explicitly for safety.
    Returns (Xw (n,2r+1,win), yw (n,), Rw (n,2r+1,4), pw (n,)).
    """
    t = 2 * radius + 1
    Xw, yw, Rw, pw = [], [], [], []
    order = np.argsort(p, kind='stable')
    Xo, yo, po = X[order], y[order], p[order]
    ro = rr[order] if rr is not None else None
    start = 0
    for i in range(1, len(po) + 1):
        if i == len(po) or po[i] != po[start]:
            run = np.arange(start, i)
            for j in range(radius, len(run) - radius):
                idx = run[j - radius:j + radius + 1]
                Xw.append(Xo[idx])
                yw.append(yo[run[j]])
                if ro is not None:
                    Rw.append(ro[idx])
                pw.append(po[run[j]])
            start = i
    return (np.stack(Xw).astype(np.float32), np.array(yw, dtype=np.int64),
            np.stack(Rw).astype(np.float32) if Rw else None,
            np.array(pw, dtype=np.int64))


def repeat_minority_windows(Xw, yw, Rw, pw, target_classes=(1, 3), seed=0):
    """Repeat real minority-center windows (train only). Sequence analogue
    of random oversampling: every training window stays fully real."""
    rng = np.random.default_rng(seed)
    ref = int(np.bincount(yw, minlength=4)[2])
    out = [[Xw], [yw], [Rw], [pw]]
    for cls in target_classes:
        m = np.where(yw == cls)[0]
        need = max(0, ref - len(m))
        if need == 0:
            continue
        picks = m[rng.integers(0, len(m), need)]
        out[0].append(Xw[picks])
        out[1].append(yw[picks])
        out[2].append(Rw[picks])
        out[3].append(pw[picks])
    return (np.concatenate(out[0]), np.concatenate(out[1]),
            np.concatenate(out[2]), np.concatenate(out[3]))


def synthetic_beats(n=12000, ratios=(0.90, 0.03, 0.06, 0.01), seed=0, patients=30):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, BEAT_LEN)
    counts = [int(n * r) for r in ratios]
    counts[0] += n - sum(counts)
    Xs, ys, ps, rs = [], [], [], []
    for cls, c in enumerate(counts):
        for _ in range(c):
            shift = rng.integers(-6, 7)
            beat = _template(cls, t)
            beat = np.roll(beat, shift)
            beat = beat + rng.normal(0, 0.12, BEAT_LEN)  # heavier noise
            beat = beat + 0.15 * np.sin(2 * np.pi * (t + rng.uniform(0, 1)))  # wander
            beat = beat * rng.uniform(0.7, 1.3)  # gain var
            Xs.append(z_norm(beat))
            ys.append(cls)
            ps.append(rng.integers(0, patients))
            # S beats are premature: short pre-RR, like real supraventricular ectopy
            pre = float(rng.normal(0.52, 0.08)) if cls == 1 else float(rng.normal(0.82, 0.10))
            post = float(rng.normal(0.82, 0.10))
            local = float(rng.normal(0.82, 0.05))
            pre, post, local = max(pre, 0.25), max(post, 0.25), max(local, 0.25)
            rs.append((pre, post, local, pre / local))
    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    p = np.array(ps, dtype=np.int64)
    rr = np.array(rs, dtype=np.float32)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], p[idx], rr[idx]
