"""Top2-weighted CNN + SeqECG logit ensemble (research #3a: cheapest macro bump).

Both checkpoints are evaluated on the SAME windowed test set: the CNN scores
the center beat of each window, SeqECG scores the whole window. Logits are
combined as w_cnn * z_cnn + w_seq * z_seq. No training, no leakage.
"""
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import recall_score, accuracy_score

from src.data import synthetic_beats, extract_beats_wfdb, make_windows, CLASS_NAMES
from src.model import EfficientECG, SeqECG
from src.train import get_data


def load_model(path):
    z = torch.load(path, map_location='cpu', weights_only=False)
    meta = z['meta']
    if meta['arch'] == 'cnn':
        m = EfficientECG()
    elif meta['arch'] == 'seq':
        m = SeqECG(use_adv=meta.get('use_adv', False),
                   n_patients=meta.get('n_patients', 0))
    else:
        raise ValueError(f"unknown arch {meta['arch']}")
    m.load_state_dict(z['state_dict'])
    return m.eval()


def ensemble_logits(m_cnn, m_seq, Xw, Rw, bs=256):
    """Weighted-sum hookable at call time; default equal weights here."""
    dl = DataLoader(TensorDataset(torch.from_numpy(Xw), torch.from_numpy(Rw)),
                    batch_size=bs)
    out = []
    with torch.no_grad():
        for xb, rb in dl:
            c = xb.shape[1] // 2
            z_cnn = m_cnn(xb[:, c].unsqueeze(1), rb[:, c])
            o = m_seq(xb, rb)
            z_seq = o[0] if isinstance(o, tuple) else o
            out.append((z_cnn, z_seq))
    return torch.cat([a for a, _ in out]), torch.cat([b for _, b in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cnn', required=True)
    ap.add_argument('--seq', required=True)
    ap.add_argument('--w-cnn', type=float, default=0.5)
    ap.add_argument('--w-seq', type=float, default=0.5)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--patient-split', action='store_true')
    ap.add_argument('--radius', type=int, default=4)
    a = ap.parse_args()
    mode = 'real' if a.real else 'synthetic'
    (Xtr, ytr, rrtr, ptr), (Xte, yte, rrte, pte) = get_data(mode, a.patient_split)
    Xw, yw, Rw, _ = make_windows(Xte, yte, pte, rrte, radius=a.radius)
    m_cnn, m_seq = load_model(a.cnn), load_model(a.seq)
    z_cnn, z_seq = ensemble_logits(m_cnn, m_seq, Xw, Rw)
    pred = (a.w_cnn * z_cnn + a.w_seq * z_seq).argmax(1).numpy()
    pc = ' '.join(f"{c}={v:.2f}" for c, v in
                  zip(CLASS_NAMES, recall_score(yw, pred, average=None,
                                               zero_division=0)))
    print(f"[ensemble {a.w_cnn}/{a.w_seq}] acc={accuracy_score(yw, pred):.3f} "
          f"macro-recall={recall_score(yw, pred, average='macro', zero_division=0):.3f} "
          f"{pc} n={len(yw)}")


if __name__ == '__main__':
    main()
