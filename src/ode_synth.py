"""ODE-inspired F synthesis: FHN-pair N-then-V delayed blend (train only).

Physiology: a fusion beat IS near-simultaneous normal + ventricular
activation. Instead of learning F from 414 examples (GAN) or interpolating
F<->F in raw space (SMOTE), blend a real N beat with a time-shifted real V
beat at varied delays -- a 2-oscillator (FitzHugh-Nagumo-pair) view with a
controllable coupling delay. DS1-only; plausibility filter rejects
off-manifold blends (QRS/peak/correlation gates). Verified on DS2 only.
"""
import argparse
import numpy as np


def znorm(x):
    m, s = x.mean(), x.std() + 1e-8
    return ((x - m) / s).astype(np.float32)


def blend_pair(n, v, lam, delay):
    return ((1 - lam) * n + lam * np.roll(v, delay)).astype(np.float32)


def plausible(m, n_med, v_med):
    if not np.isfinite(m).all() or np.abs(m).max() > 6.0:
        return False
    pk = int(np.abs(m).argmax())
    if pk < 47 or pk > 141:  # peak must sit in the central half (QRS gate)
        return False
    cn = float(np.corrcoef(m, n_med)[0, 1])
    cv = float(np.corrcoef(m, v_med)[0, 1])
    return (cn > 0.5) or (cv > 0.5)  # must resemble at least one parent


def synthesize_f(Xn, Xv, rrf, n_gen, seed=0, lam_lo=0.3, lam_hi=0.7,
                 max_delay=16):
    """Blend random DS1 N/V pairs at random delays; resample real DS1-F RR."""
    rng = np.random.default_rng(seed)
    n_med = np.median(Xn[: min(2000, len(Xn))], axis=0)
    v_med = np.median(Xv[: min(2000, len(Xv))], axis=0)
    out, tries, kept = [], 0, 0
    while kept < n_gen and tries < n_gen * 20:
        tries += 1
        n = Xn[rng.integers(0, len(Xn))]
        v = Xv[rng.integers(0, len(Xv))]
        m = znorm(blend_pair(n, v, float(rng.uniform(lam_lo, lam_hi)),
                             int(rng.integers(-max_delay, max_delay + 1))))
        if plausible(m, n_med, v_med):
            out.append(m)
            kept += 1
    if kept < n_gen:
        print(f"warn: plausibility kept {kept}/{n_gen}", flush=True)
    Xs = np.stack(out).astype(np.float32) if out else np.zeros((0, 188), np.float32)
    rs = rrf[rng.integers(0, len(rrf), len(Xs))].astype(np.float32)
    return Xs, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-f', action='store_true')
    ap.add_argument('--n-gen', type=int, default=2000)
    ap.add_argument('--out', default='ode_f_beats.npz')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.real_f:
        from src.data import extract_beats_wfdb, DE_CHAZAL_DS1
        X, y, p, rr = extract_beats_wfdb()
        ds1 = np.isin(p, DE_CHAZAL_DS1)
        Xn, Xv = X[ds1 & (y == 0)], X[ds1 & (y == 2)]
        rrf = rr[ds1 & (y == 3)]
        print(f"DS1 N={len(Xn)} V={len(Xv)} F={len(rrf)}", flush=True)
    else:
        from src.data import synthetic_beats
        X, y, p, rr = synthetic_beats()
        Xn, Xv = X[y == 0], X[y == 2]
        rrf = rr[y == 3]
    Xs, rs = synthesize_f(Xn, Xv, rrf, a.n_gen, seed=a.seed)
    np.savez(a.out, X=Xs, rr=rs)
    print(f"saved {len(Xs)} ODE-blend F beats -> {a.out}")


if __name__ == '__main__':
    main()
