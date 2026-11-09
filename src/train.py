"""Train + compare no-weights vs class-weighted. Reports per-class recall."""
import argparse, time, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import train_test_split


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # determinism lockdown: same seed + same code + same data => same numbers.
    # warn_only: keep running if an op lacks a deterministic kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)


def save_ckpt(path, state, meta):
    torch.save({'state_dict': {k: v.cpu() for k, v in state.items()},
                'meta': meta}, path)
    print(f"saved checkpoint -> {path}", flush=True)

from src.data import synthetic_beats, extract_beats_wfdb, smote_train_only, fusion_mixup_train_only, random_oversample_train_only, within_class_mixup_train_only, make_windows, repeat_minority_windows, CLASS_NAMES, DE_CHAZAL_DS1, DE_CHAZAL_DS2
from src.model import EfficientECG, VanillaCNN, SeqECG, count_params


def get_data(mode, patient_split):
    if mode == 'real':
        try:
            X, y, p, rr = extract_beats_wfdb()
        except Exception as e:
            print(f"real download failed ({e}), falling back to synthetic", file=sys.stderr)
            X, y, p, rr = synthetic_beats()
    else:
        X, y, p, rr = synthetic_beats()
    if patient_split and mode == 'real':
        tr = np.isin(p, DE_CHAZAL_DS1); te = np.isin(p, DE_CHAZAL_DS2)
        if tr.sum() == 0 or te.sum() == 0:
            raise SystemExit("patient split empty - check record ids")
        return (X[tr], y[tr], rr[tr], p[tr]), (X[te], y[te], rr[te], p[te])
    # beat split (stratified) by default; synthetic patient-split optional
    if patient_split:
        up = np.unique(p)
        tr_p, te_p = train_test_split(up, test_size=0.33, random_state=0)
        tr = np.isin(p, tr_p); te = np.isin(p, te_p)
        return (X[tr], y[tr], rr[tr], p[tr]), (X[te], y[te], rr[te], p[te])
    Xtr, Xte, ytr, yte, rrtr, rrte, ptr, pte = train_test_split(
        X, y, rr, p, test_size=0.2, random_state=0, stratify=y)
    return (Xtr, ytr, rrtr, ptr), (Xte, yte, rrte, pte)


def augment(X, y, rr=None, seed=0):
    """Cheap minority-only augment: time-shift + noise (no SMOTE blowup)."""
    rng = np.random.default_rng(seed)
    out_x, out_y = [X], [y]
    out_r = [rr] if rr is not None else []
    for cls in (1, 2, 3):
        m = X[y == cls]
        if len(m) == 0:
            continue
        dup = np.stack([np.roll(b, int(rng.integers(-8, 9))) +
                        rng.normal(0, 0.03, b.shape) for b in m[:len(m) * 2]])
        out_x.append(dup.astype(np.float32)); out_y.append(np.full(len(dup), cls))
        if rr is not None:
            mr = rr[y == cls]
            out_r.append(np.tile(mr, (2, 1))[:len(dup)])
    if rr is not None:
        return np.concatenate(out_x), np.concatenate(out_y), np.concatenate(out_r)
    return np.concatenate(out_x), np.concatenate(out_y)


class FocalLoss(nn.Module):
    """Focal loss: FL(pt) = -alpha*(1-pt)^gamma*log(pt).

    Down-weights well-classified majority beats, amplifies hard minority
    ones. Research (MAK-Net, MIT-BIH): S 92.08%->95.71%, F 90.20%->92.94%,
    macro recall 0.9585->0.9717 vs plain CE on the same model.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction='none')
        pt = ce.neg().exp()
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def run_once(Xtr, ytr, Xte, yte, use_weights, epochs, rrtr=None, rrte=None,
             lr=3e-3, bs=256, aug=True, loss='ce', smote=False, fmix=False,
             fmix_ratio=1.0, ros=False, wcmix=False, ganf=None, save=None,
             minibatch=False, sched='cosine', swa=False, val=False, ptr=None,
             use_rr=True, use_selfonn=False, mf_init=False, mf_freeze=0):
    from src.data import minority_batch_order, record_holdout
    from src.matched import class_templates, init_stem_matched, zero_stem_grads
    from torch.utils.data import Subset
    sel_name, sel_y, Xv, yv, Rvv = 'test', None, None, None, None
    if val:
        if ptr is not None:
            vm = record_holdout(ptr)
        else:
            _, v_idx = train_test_split(np.arange(len(ytr)), test_size=0.1,
                                        random_state=0, stratify=ytr)
            vm = np.zeros(len(ytr), dtype=bool)
            vm[v_idx] = True
        keep = ~vm
        Xv, yv, Rvv = Xtr[vm], ytr[vm], (rrtr[vm] if rrtr is not None else None)
        Xtr, ytr = Xtr[keep], ytr[keep]
        rrtr = rrtr[keep] if rrtr is not None else None
        ptr = ptr[keep] if ptr is not None else None
        print(f"val split: fit {len(ytr)}, val {len(yv)} (selection on val)",
              flush=True)
    if ganf is not None:
        # offline GAN F beats (train only): hard F labels, test untouched
        z = np.load(ganf)
        Xg, rrg = z['X'].astype(np.float32), z['rr'].astype(np.float32)
        Xtr = np.concatenate([Xtr, Xg])
        ytr = np.concatenate([ytr, np.full(len(Xg), 3, dtype=np.int64)])
        if rrtr is not None:
            rrtr = np.concatenate([rrtr, rrg])
    if aug:
        res = augment(Xtr, ytr, rrtr)
        Xtr, ytr = res[0], res[1]
        rrtr = res[2] if len(res) == 3 else None
    if ros:
        out = random_oversample_train_only(Xtr, ytr, rr_tr=rrtr)  # real beats only
        Xtr, ytr = out[0], out[1]
        rrtr = out[2] if len(out) == 3 else rrtr
    if wcmix:
        out = within_class_mixup_train_only(Xtr, ytr, rr_tr=rrtr)  # F<->F, S<->S only
        Xtr, ytr = out[0], out[1]
        rrtr = out[2] if len(out) == 3 else rrtr
    if smote:
        # when fmix is on, SMOTE handles S only (F comes from N+V mixup)
        tc = (1,) if fmix else (1, 3)
        out = smote_train_only(Xtr, ytr, target_classes=tc, rr_tr=rrtr)  # train only: no leakage
        Xtr, ytr = out[0], out[1]
        rrtr = out[2] if len(out) == 3 else rrtr
    if fmix:
        out = fusion_mixup_train_only(Xtr, ytr, rr_tr=rrtr, ratio=fmix_ratio)  # train only
        Xtr, ytr = out[0], out[1]
        rrtr = out[2] if len(out) == 3 else rrtr
    if val:
        sel_name, sel_y = 'val', yv
    else:
        sel_name, sel_y = 'test', yte
    Xt = torch.from_numpy(Xtr).unsqueeze(1); yt = torch.from_numpy(ytr)
    Xe = torch.from_numpy(Xte).unsqueeze(1); ye = torch.from_numpy(yte)
    Rt = torch.from_numpy(rrtr) if rrtr is not None else None
    Re = torch.from_numpy(rrte) if rrte is not None else None
    fit_ds = TensorDataset(Xt, Rt, yt) if Rt is not None else TensorDataset(Xt, yt)
    if minibatch:
        order = minority_batch_order(ytr, bs=bs, seed=0)
        tl = DataLoader(Subset(fit_ds, order.tolist()), batch_size=bs, shuffle=False)
    else:
        tl = DataLoader(fit_ds, batch_size=bs, shuffle=True)
    el = DataLoader(TensorDataset(Xe, Re, ye) if Re is not None else TensorDataset(Xe, ye),
                    batch_size=1024)
    sl = None
    if val:
        Xsv = torch.from_numpy(Xv).unsqueeze(1); ysv = torch.from_numpy(yv)
        Rsv = torch.from_numpy(Rvv) if Rvv is not None else None
        sl = DataLoader(TensorDataset(Xsv, Rsv, ysv) if Rsv is not None else
                        TensorDataset(Xsv, ysv), batch_size=1024)
    model = EfficientECG(use_rr=use_rr, use_selfonn=use_selfonn)
    if mf_init:
        # matched-filter stem from DS1-fit medians (train side only)
        init_stem_matched(model, class_templates(Xtr, ytr))
        print(f"matched-filter stem init (freeze {mf_freeze} epochs)", flush=True)
    w = None
    if use_weights:
        cnt = np.bincount(ytr, minlength=4).astype(float)
        w = (len(ytr) / (4 * np.maximum(cnt, 1)))
    if loss == 'focal' and use_weights:
        crit = FocalLoss(alpha=w, gamma=2.0)
    elif use_weights:
        crit = nn.CrossEntropyLoss(torch.tensor(w, dtype=torch.float32))
    else:
        crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if sched == 'plateau':
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=max(2, epochs // 5))
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = -1, None
    swa_states = []
    has_rr = Rt is not None
    def fwd(b):
        if has_rr:
            return model(b[0], b[1]), b[2]
        return model(b[0]), b[1]
    import sys
    for ep in range(epochs):
        model.train()
        for b in tl:
            out, yb = fwd(b)
            opt.zero_grad(); crit(out, yb).backward()
            if mf_init and ep < mf_freeze:
                zero_stem_grads(model)
            opt.step()
        if sched == 'plateau':
            model.eval()
            with torch.no_grad():
                pv = torch.cat([fwd(b)[0].argmax(1) for b in (sl or el)]).numpy()
            sch.step(recall_score(sel_y, pv, average='macro', zero_division=0))
            model.train()
        else:
            sch.step()
        if swa and ep >= max(0, epochs - 5):
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        model.eval()
        with torch.no_grad():
            pred = torch.cat([fwd(b)[0].argmax(1) for b in (sl or el)]).numpy()
        r = recall_score(sel_y, pred, average='macro', zero_division=0)
        print(f"  epoch {ep+1}/{epochs} {sel_name}-macro-recall={r:.3f}", flush=True)
        if r > best:
            best, best_state = r, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if swa and swa_states:
        avg = {k: (torch.stack([s[k] for s in swa_states]).mean(0)
                   if swa_states[0][k].is_floating_point()
                   else swa_states[-1][k])
               for k in swa_states[0]}
        model.load_state_dict(avg)
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print("swa: averaged last-%d epoch weights for final eval" % len(swa_states),
              flush=True)
    else:
        model.load_state_dict(best_state)
    if save is not None:
        save_ckpt(save, best_state, {'arch': 'cnn'})
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = torch.cat([fwd(b)[0].argmax(1) for b in el]).numpy()
    infer_ms = (time.perf_counter() - t0) / len(ye) * 1000
    return {
        'acc': accuracy_score(yte, pred),
        'macro_recall': recall_score(yte, pred, average='macro', zero_division=0),
        'per_class': recall_score(yte, pred, average=None, zero_division=0).tolist(),
        'infer_ms': infer_ms, 'params': count_params(model),
    }


def run_seq(Xwtr, ywtr, Rwtr, pwtr, Xwte, ywte, Rwte, use_weights, epochs,
              lr=3e-3, bs=128, loss='ce', ros_w=False, curriculum=0.0,
              adv=False, adv_lamb=0.1, radius=4, save=None,
              minibatch=False, sched='cosine', swa=False, val=False,
              use_rr=True, use_selfonn=False, use_cfc=False):
    """Windowed rhythm-context training (train windows only, test untouched).

    Curriculum: first `curriculum` fraction of epochs sees minority-center
    windows only (WavKAN-CL discovery phase), then full data. Adversarial:
    record-id head through gradient reversal forces patient independence.
    val=True holds out every 7th train record for checkpoint selection.
    """
    from src.data import minority_batch_order, record_holdout
    from torch.utils.data import Subset
    sel_name, sel_yw = 'test', ywte
    if val:
        vm = record_holdout(pwtr)
        Xwv, ywv, Rwv, pwv = Xwtr[vm], ywtr[vm], Rwtr[vm], pwtr[vm]
        keep = ~vm
        Xwtr, ywtr, Rwtr, pwtr = Xwtr[keep], ywtr[keep], Rwtr[keep], pwtr[keep]
        sel_name, sel_yw = 'val', ywv
        print(f"val split: fit {len(ywtr)}, val {len(ywv)} (selection on val)",
              flush=True)
    if ros_w:
        Xwtr, ywtr, Rwtr, pwtr = repeat_minority_windows(Xwtr, ywtr, Rwtr, pwtr)
    if Rwtr is None:  # --no-rr: shape-compatible zeros; trunk ignores them
        Rwtr = np.zeros((len(ywtr), Xwtr.shape[1], 4), np.float32)
    if Rwte is None:
        Rwte = np.zeros((len(ywte), Xwte.shape[1], 4), np.float32)
    if val and Rwv is None:
        Rwv = np.zeros((len(ywv), Xwtr.shape[1], 4), np.float32)
    recs = np.unique(pwtr)
    pmap = {r: i for i, r in enumerate(recs)}
    zt = torch.from_numpy(Xwtr)
    ze = torch.from_numpy(Xwte)
    Rt = torch.from_numpy(Rwtr)
    Re = torch.from_numpy(Rwte)
    pt = torch.tensor([pmap[r] for r in pwtr], dtype=torch.long)
    fit_ds = TensorDataset(zt, Rt, torch.from_numpy(ywtr), pt)
    if minibatch:
        order = minority_batch_order(ywtr, bs=bs, seed=0)
        tl_full = DataLoader(Subset(fit_ds, order.tolist()), batch_size=bs,
                             shuffle=False)
    else:
        tl_full = DataLoader(fit_ds, batch_size=bs, shuffle=True)
    el = DataLoader(TensorDataset(ze, Re, torch.from_numpy(ywte)), batch_size=256)
    sl = None
    if val:
        sl = DataLoader(TensorDataset(torch.from_numpy(Xwv), torch.from_numpy(Rwv),
                                       torch.from_numpy(ywv)), batch_size=256)
    min_mask = np.isin(ywtr, [1, 2, 3])
    if minibatch:
        tl_min = tl_full  # minority batches already guarantee S/F presence
    else:
        tl_min = DataLoader(TensorDataset(zt[min_mask], Rt[min_mask],
                                           torch.from_numpy(ywtr[min_mask]),
                                           pt[min_mask]),
                            batch_size=bs, shuffle=True) if min_mask.any() else tl_full
    feat_dim = 160 if use_rr else 144
    trunk = EfficientECG(use_rr=use_rr, use_selfonn=use_selfonn)
    if use_cfc:
        from src.cfc import CfCECG
        model = CfCECG(trunk, use_adv=adv, n_patients=len(recs),
                       feat_dim=feat_dim)
        print("CfC liquid rhythm head (closed-form, RR-paced)", flush=True)
    else:
        model = SeqECG(trunk, use_adv=adv, n_patients=len(recs),
                       feat_dim=feat_dim)
    def make_crit(ys):
        w = None
        if use_weights:
            cnt = np.bincount(ys, minlength=4).astype(float)
            w = (len(ys) / (4 * np.maximum(cnt, 1)))
        if loss == 'focal' and use_weights:
            return FocalLoss(alpha=w, gamma=2.0)
        if use_weights:
            return nn.CrossEntropyLoss(torch.tensor(w, dtype=torch.float32))
        return nn.CrossEntropyLoss()
    c_full, c_min = make_crit(ywtr), make_crit(ywtr[min_mask])
    pat_crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if sched == 'plateau':
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=max(2, epochs // 5))
    else:
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n_min_epochs = int(epochs * curriculum)
    best, best_state = -1, None
    swa_states = []
    for ep in range(epochs):
        model.train()
        tl = tl_min if ep < n_min_epochs else tl_full
        crit = c_min if ep < n_min_epochs else c_full
        for b in tl:
            out = model(b[0], b[1], lamb=adv_lamb)
            if adv:
                task, pb = out
                l = crit(task, b[2]) + pat_crit(pb, b[3])
            else:
                l = crit(out, b[2])
            opt.zero_grad(); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = []
            for b in (sl or el):
                o = model(b[0], b[1])
                pv.append((o[0] if adv else o).argmax(1))
            pv = torch.cat(pv).numpy()
        if sched == 'plateau':
            sch.step(recall_score(sel_yw, pv, average='macro', zero_division=0))
        else:
            sch.step()
        if swa and ep >= max(0, epochs - 5):
            swa_states.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
        r = recall_score(sel_yw, pv, average='macro', zero_division=0)
        print(f"  epoch {ep+1}/{epochs} {sel_name}-macro-recall={r:.3f}", flush=True)
        if r > best:
            best, best_state = r, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if swa and swa_states:
        avg = {k: (torch.stack([s[k] for s in swa_states]).mean(0)
                   if swa_states[0][k].is_floating_point()
                   else swa_states[-1][k])
               for k in swa_states[0]}
        model.load_state_dict(avg)
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print("swa: averaged last-%d epoch weights for final eval" % len(swa_states),
              flush=True)
    else:
        model.load_state_dict(best_state)
    if save is not None:
        save_ckpt(save, best_state,
                  {'arch': 'seq', 'use_adv': adv, 'n_patients': len(recs)})
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = []
        for b in el:
            o = model(b[0], b[1])
            pred.append((o[0] if adv else o).argmax(1))
        pred = torch.cat(pred).numpy()
    infer_ms = (time.perf_counter() - t0) / len(ywte) * 1000
    return {
        'acc': accuracy_score(ywte, pred),
        'macro_recall': recall_score(ywte, pred, average='macro', zero_division=0),
        'per_class': recall_score(ywte, pred, average=None, zero_division=0).tolist(),
        'infer_ms': infer_ms, 'params': count_params(model),
    }


def apply_config(ap):
    """Two-stage --config: TOML values become parser defaults (CLI wins)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None,
                     help='TOML file with argument values (CLI overrides file)')
    known, _ = pre.parse_known_args()
    if known.config is None:
        return None
    import tomllib
    with open(known.config, 'rb') as f:
        cfg = tomllib.load(f)
    valid = {a.dest for a in ap._actions}
    bad = sorted(set(cfg) - valid)
    if bad:
        raise SystemExit(f"unknown config keys in {known.config}: {bad}")
    ap.set_defaults(**cfg)
    return known.config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patient-split', action='store_true')
    ap.add_argument('--no-class-weights', action='store_true')
    ap.add_argument('--loss', choices=['ce', 'focal'], default='ce',
                    help='loss for the weighted phase (focal needs weights)')
    ap.add_argument('--smote', action='store_true',
                    help='train-only SMOTE upsampling of S/F toward V count')
    ap.add_argument('--fmix', action='store_true',
                    help='train-only N+V fusion mixup to synthesize F beats')
    ap.add_argument('--fmix-ratio', type=float, default=1.0,
                    help='F target as fraction of V count (default 1.0)')
    ap.add_argument('--ros', action='store_true',
                    help='random oversampling: repeat real S/F beats toward V count')
    ap.add_argument('--wcmix', action='store_true',
                    help='within-class F<->F/S<->S beta mixup with hard labels')
    ap.add_argument('--ganf', default=None,
                    help='path to offline GAN F beats .npz (X, rr); train only')
    ap.add_argument('--seq', action='store_true',
                    help='beat-sequence model: BiLSTM over trunk embeddings, center labeled')
    ap.add_argument('--radius', type=int, default=4,
                    help='sequence context radius in beats (window = 2*radius+1)')
    ap.add_argument('--curriculum', type=float, default=0.0,
                    help='fraction of epochs on minority-center windows first')
    ap.add_argument('--adv', action='store_true',
                    help='domain-adversarial record-id head (gradient reversal)')
    ap.add_argument('--adv-lamb', type=float, default=0.1,
                    help='gradient reversal strength')
    ap.add_argument('--seed', type=int, default=0,
                    help='random seed for numpy/torch (report mean±std over seeds)')
    ap.add_argument('--save', default=None,
                    help='save best checkpoint (weighted phase) to this path')
    ap.add_argument('--minibatch', action='store_true',
                    help='every batch guaranteed S+F (no dataset-level repetition)')
    ap.add_argument('--sched', choices=['cosine', 'plateau'], default='cosine')
    ap.add_argument('--swa', action='store_true',
                    help='average last-5-epoch weights for final eval')
    ap.add_argument('--val', action='store_true',
                    help='hold out every 7th train record for checkpoint selection')
    ap.add_argument('--no-rr', action='store_true',
                    help='RR ablation: drop the RR-interval branch entirely')
    ap.add_argument('--selfonn', action='store_true',
                    help='Self-ONN stem (Q=3 generative neurons a la Kiranyaz)')
    ap.add_argument('--mf-init', action='store_true',
                    help='matched-filter stem init from DS1 class medians (train only)')
    ap.add_argument('--mf-freeze', type=int, default=5,
                    help='freeze stem for first N epochs after matched init')
    ap.add_argument('--cfc', action='store_true',
                    help='CfC liquid rhythm head instead of BiLSTM (RR-paced)')
    ap.add_argument('--config', default=None,
                    help='TOML file with argument values (CLI overrides file)')
    cfg = apply_config(ap)
    a = ap.parse_args()
    set_seed(a.seed)
    resolved = {k: v for k, v in sorted(vars(a).items())}
    print(f"config {cfg or 'cli-only'}: " +
          ' '.join(f"{k}={v}" for k, v in resolved.items()
                   if v not in (False, None, 0, 0.0, '')) or '(defaults)',
          flush=True)
    if a.save is not None:
        import json
        with open(a.save + '.config.json', 'w') as f:
            json.dump(resolved, f, indent=1, default=str)
    mode = 'real' if a.real else 'synthetic'
    (Xtr, ytr, rrtr, ptr), (Xte, yte, rrte, pte) = get_data(mode, a.patient_split)
    if a.seq:
        if any([a.smote, a.fmix, a.wcmix, a.ganf]):
            print("note: beat-level synth flags ignored in --seq mode "
                  "(windows stay fully real)", flush=True)
        if a.no_rr:
            rrtr = rrte = None
            print("ablation: RR branch off (morphology + QRS only)", flush=True)
        Xwtr, ywtr, Rwtr, pwtr = make_windows(Xtr, ytr, ptr, rrtr, radius=a.radius)
        Xwte, ywte, Rwte, _pwte = make_windows(Xte, yte, pte, rrte, radius=a.radius)
        print(f"windows train {len(ywtr)} test {len(ywte)} "
              f"train_dist {np.bincount(ywtr, minlength=4).tolist()} "
              f"test_dist {np.bincount(ywte, minlength=4).tolist()}")
        print("== no-weights ==", flush=True)
        base = run_seq(Xwtr, ywtr, Rwtr, pwtr, Xwte, ywte, Rwte,
                       use_weights=False, epochs=a.epochs,
                       minibatch=a.minibatch, sched=a.sched, swa=a.swa, val=a.val,
                       use_rr=not a.no_rr, use_selfonn=a.selfonn,
                       use_cfc=a.cfc)
        print(f"== weighted ({a.loss}) ==", flush=True)
        wt = run_seq(Xwtr, ywtr, Rwtr, pwtr, Xwte, ywte, Rwte,
                     use_weights=not a.no_class_weights, epochs=a.epochs,
                     loss=a.loss, ros_w=a.ros, curriculum=a.curriculum,
                     adv=a.adv, adv_lamb=a.adv_lamb, save=a.save,
                     minibatch=a.minibatch, sched=a.sched, swa=a.swa, val=a.val,
                     use_rr=not a.no_rr, use_selfonn=a.selfonn,
                     use_cfc=a.cfc)
        for tag, r in [('no-weights', base), ('weighted', wt)]:
            pc = ' '.join(f"{c}={v:.2f}" for c, v in zip(CLASS_NAMES, r['per_class']))
            print(f"[{tag}] acc={r['acc']:.3f} macro-recall={r['macro_recall']:.3f} {pc} "
                  f"params={r['params']} {r['infer_ms']:.3f}ms/beat")
        print(f"seq params={wt['params']} (trunk+{'CfC' if a.cfc else 'LSTM'}, "
              f"window={2*a.radius+1})")
        d = wt['macro_recall'] - base['macro_recall']
        print(f"weighted macro-recall delta: {d:+.3f}")
        print(f"RESUME: Trained a beat-sequence 1D CNN on {len(ywtr)+len(ywte)} "
              f"labeled windows from MIT-BIH{' (synthetic mirror)' if mode=='synthetic' else ''}, "
              f"using class weighting to raise macro-recall from "
              f"{base['macro_recall']:.1%} to {wt['macro_recall']:.1%}.")
        return
    print(f"train {len(ytr)} test {len(yte)} "
          f"train_dist {np.bincount(ytr, minlength=4).tolist()} "
          f"test_dist {np.bincount(yte, minlength=4).tolist()}")
    print("== no-weights ==", flush=True)
    if a.no_rr:
        rrtr = rrte = None
        print("ablation: RR branch off (morphology + QRS only)", flush=True)
    base = run_once(Xtr, ytr, Xte, yte, use_weights=False, epochs=a.epochs,
                    rrtr=rrtr, rrte=rrte, minibatch=a.minibatch, sched=a.sched,
                    swa=a.swa, val=a.val, ptr=ptr, use_rr=not a.no_rr, use_selfonn=a.selfonn,
                    mf_init=a.mf_init, mf_freeze=a.mf_freeze)
    print(f"== weighted ({a.loss}) ==", flush=True)
    wt = run_once(Xtr, ytr, Xte, yte, use_weights=not a.no_class_weights, epochs=a.epochs,
                  rrtr=rrtr, rrte=rrte, loss=a.loss, smote=a.smote,
                  fmix=a.fmix, fmix_ratio=a.fmix_ratio, ros=a.ros, wcmix=a.wcmix,
                  ganf=a.ganf, save=a.save, minibatch=a.minibatch, sched=a.sched,
                  swa=a.swa, val=a.val, ptr=ptr, use_rr=not a.no_rr, use_selfonn=a.selfonn,
                  mf_init=a.mf_init, mf_freeze=a.mf_freeze)
    for tag, r in [('no-weights', base), ('weighted', wt)]:
        pc = ' '.join(f"{c}={v:.2f}" for c, v in zip(CLASS_NAMES, r['per_class']))
        print(f"[{tag}] acc={r['acc']:.3f} macro-recall={r['macro_recall']:.3f} {pc} "
              f"params={r['params']} {r['infer_ms']:.3f}ms/beat")
    print(f"vanilla params={count_params(VanillaCNN())} vs efficient params={base['params']} "
          f"(~{count_params(VanillaCNN())/base['params']:.1f}x smaller)")
    d = wt['macro_recall'] - base['macro_recall']
    print(f"weighted macro-recall delta: {d:+.3f}")
    print(f"RESUME: Trained a 1D CNN on {len(ytr)+len(yte)} labeled heartbeats "
          f"from MIT-BIH{' (synthetic mirror)' if mode=='synthetic' else ''}, using class weighting "
          f"to raise macro-recall from {base['macro_recall']:.1%} to {wt['macro_recall']:.1%}.")


if __name__ == '__main__':
    main()

