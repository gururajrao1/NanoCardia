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


# Canonical PQRST angles (R at 0) and widths; amplitudes fit to DS1-F.
ECG_ANGLES = np.array([-1.9, -0.35, 0.0, 0.35, 1.6])
ECG_WIDTHS = np.array([0.25, 0.10, 0.12, 0.12, 0.30])


def _wrap(dtheta):
    return (np.asarray(dtheta) + np.pi) % (2 * np.pi) - np.pi


def fit_gaussians(med, angles=ECG_ANGLES, widths=ECG_WIDTHS, crop=128):
    """Least-squares PQRST amplitudes of a median beat (linear in a_i).

    Central `crop` samples around the R peak map to one [-pi, pi] cycle;
    baseline z0 comes from the window edges.
    """
    med = np.asarray(med, dtype=np.float64)
    L = len(med)
    r = int(np.abs(med).argmax())
    lo, hi = max(0, r - crop // 2), min(L, r + crop // 2)
    seg = med[lo:hi]
    th = (np.arange(len(seg)) - (r - lo)) / len(seg) * 2 * np.pi
    G = np.exp(-_wrap(th[:, None] - angles[None, :]) ** 2 /
               (2 * widths[None, :] ** 2))
    z0 = float(np.median(np.concatenate([med[:10], med[-10:]])))
    a, _, _, _ = np.linalg.lstsq(G, seg - z0, rcond=None)
    return a.astype(np.float32), float(z0)


def _ecg_cycle(a, z0, angles, widths, cycle_len, steps_per_cycle=600):
    """Integrate McSharry-Clifford ODEs for 2 cycles (Euler, 600 st/cyc).

    Theta advances uniformly (verified std ~4e-5), so R crossings sit at
    spc/2 and 3spc/2; the crop centers on the last interior one.
    """
    import math
    omega = 2 * math.pi / cycle_len
    dt = cycle_len / steps_per_cycle
    n_steps = int(2.0 * steps_per_cycle)
    x, y, z = -1.0, 0.0, z0
    zs = np.empty(n_steps)
    for t in range(n_steps):
        th = math.atan2(y, x)
        dth = _wrap(th - angles)
        # McSharry Eq.3: derivative-of-Gaussian forcing (note /b^2)
        dz = float(-(a * dth / widths ** 2 *
                     np.exp(-dth ** 2 / (2 * widths ** 2))).sum())
        alpha = 1 - math.hypot(x, y)
        x += (alpha * x - omega * y) * dt
        y += (alpha * y + omega * x) * dt
        z += (dz - (z - z0)) * dt
        zs[t] = z
    return zs


def _crop_beat(zs, steps_per_cycle, cycle_len, out_len=188, fs=125.0):
    """Center a 1.5s window on the last R crossing, resample to out_len."""
    n = len(zs)
    # R sits exactly at 3/4 through (theta advances uniformly; a/b only
    # shape z, never the phase), so no argmax hunting for it.
    r = n * 3 // 4
    sps = steps_per_cycle / cycle_len  # integration steps per second
    half = int(1.504 / 2 * sps)
    lo, hi = r - half, r + half
    pad_l, pad_r = max(0, -lo), max(0, hi - n)
    seg = np.pad(zs[max(0, lo):min(n, hi)], (pad_l, pad_r), mode='edge')
    return np.interp(np.linspace(0, len(seg) - 1, out_len),
                     np.arange(len(seg)), seg).astype(np.float32)


def synthesize_f_ecgsyn(Xf, rrf, n_gen, n_med, v_med, seed=0,
                        steps_per_cycle=600):
    """ECGSYN F beats: ODE with DS1-F-fitted Gaussians, perturbed per beat.

    Timing comes from real DS1-F RR rows (omega = 2pi/cycle), morphology
    from perturbed (theta, a, b). Gate identical to the FHN blend.
    DS1-only; verified on DS2 only.
    """
    rng = np.random.default_rng(seed)
    med = np.median(Xf[: min(2000, len(Xf))], axis=0)
    a0, z0 = fit_gaussians(med)
    out, tries, kept = [], 0, 0
    while kept < n_gen and tries < n_gen * 20:
        tries += 1
        rr = rrf[rng.integers(0, len(rrf))]
        cycle = float(np.clip((rr[0] + rr[1]) / 2, 0.5, 1.5))
        da = rng.uniform(-0.15, 0.15, size=5)
        da[2] = rng.uniform(-0.05, 0.05)  # R stays put
        a = a0 * np.exp(rng.normal(0, 0.15, size=5)).astype(np.float32)
        b = (ECG_WIDTHS * rng.uniform(0.85, 1.2, size=5)).astype(np.float32)
        zs = _ecg_cycle(a, z0 + float(rng.normal(0, 0.02)),
                        ECG_ANGLES + da, b, cycle,
                        steps_per_cycle=steps_per_cycle)
        m = znorm(_crop_beat(zs, steps_per_cycle, cycle))
        if plausible(m, n_med, v_med):
            out.append(m)
            kept += 1
    if kept < n_gen:
        print(f"warn: ecgsyn plausibility kept {kept}/{n_gen}", flush=True)
    Xs = np.stack(out).astype(np.float32) if out else np.zeros((0, 188), np.float32)
    rs = rrf[rng.integers(0, len(rrf), len(Xs))].astype(np.float32)
    return Xs, rs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-f', action='store_true')
    ap.add_argument('--method', choices=['blend', 'ecgsyn'], default='blend')
    ap.add_argument('--n-gen', type=int, default=2000)
    ap.add_argument('--out', default='ode_f_beats.npz')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    from src.data import DE_CHAZAL_DS1
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
    if a.method == 'ecgsyn':
        ds1mask = np.isin(p, DE_CHAZAL_DS1) if a.real_f else np.ones(len(y), bool)
        Xf = X[ds1mask & (y == 3)]
        n_med = np.median(Xn[: min(2000, len(Xn))], axis=0)
        v_med = np.median(Xv[: min(2000, len(Xv))], axis=0)
        Xs, rs = synthesize_f_ecgsyn(Xf, rrf, a.n_gen, n_med, v_med,
                                     seed=a.seed)
    else:
        Xs, rs = synthesize_f(Xn, Xv, rrf, a.n_gen, seed=a.seed)
    np.savez(a.out, X=Xs, rr=rs)
    print(f"saved {len(Xs)} {a.method} F beats -> {a.out}")


if __name__ == '__main__':
    main()
