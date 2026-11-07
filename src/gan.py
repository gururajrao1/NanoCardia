"""Tiny unconditional WGAN-GP for F (fusion) beats, 1D raw waveforms.

Research recipe (Adib et al., arXiv:2202.00569): per-class UNCONDITIONAL GAN on
raw unscreened beats — conditional GAN and DTW-screened variants underperform
(diversity beats fidelity; explains our SMOTE failure). Train on DS1-F only
(414 beats), generate F to ~2-5k, retrain unchanged classifier.

Guardrails: DS1-only training, hard F labels, real-F-recall gate on DS2,
drop entirely if DS2-F doesn't move.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn


class Generator(nn.Module):
    """z(64) -> (16,47) -> (8,94) -> (1,188). ~2k params."""

    def __init__(self, z_dim=64):
        super().__init__()
        self.fc = nn.Linear(z_dim, 16 * 47)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 4, stride=2, padding=1),  # 47 -> 94
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(8, 1, 4, stride=2, padding=1),    # 94 -> 188
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 16, 47))


class Critic(nn.Module):
    """(1,188) -> score. ~5k params."""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1))

    def forward(self, x):
        return self.f(x)


def gradient_penalty(critic, real, fake, device):
    b = real.size(0)
    eps = torch.rand(b, 1, 1, device=device)
    inter = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = critic(inter)
    grad = torch.autograd.grad(score, inter, torch.ones_like(score),
                               create_graph=True)[0]
    return ((grad.view(b, -1).norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(X_real, steps=2000, bs=64, z_dim=64, lr=1e-4,
                  n_critic=5, lamb=10.0, seed=0, log_every=500):
    """Train unconditional WGAN-GP on one class. Returns fitted Generator."""
    torch.manual_seed(seed)
    device = torch.device('cpu')
    gen, critic = Generator(z_dim).to(device), Critic().to(device)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    real_all = torch.from_numpy(X_real.astype(np.float32)).unsqueeze(1)
    n = len(real_all)
    idx = torch.randperm(n)
    ptr = 0
    for step in range(1, steps + 1):
        for _ in range(n_critic):
            if ptr + bs > n:
                idx = torch.randperm(n)
                ptr = 0
            real = real_all[idx[ptr:ptr + bs]].to(device)
            ptr += bs
            z = torch.randn(len(real), z_dim, device=device)
            with torch.no_grad():
                fake = gen(z)
            opt_c.zero_grad()
            loss_c = critic(fake).mean() - critic(real).mean() + \
                lamb * gradient_penalty(critic, real, fake, device)
            loss_c.backward()
            opt_c.step()
        z = torch.randn(bs, z_dim, device=device)
        opt_g.zero_grad()
        loss_g = -critic(gen(z)).mean()
        loss_g.backward()
        opt_g.step()
        if step % log_every == 0:
            print(f"  step {step}/{steps} loss_c={loss_c.item():.3f} "
                  f"loss_g={loss_g.item():.3f}", flush=True)
    return gen


def generate_beats(gen, n, z_dim=64, seed=1):
    gen.eval()
    out = []
    rng_n = 0
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        while rng_n < n:
            b = min(512, n - rng_n)
            z = torch.randn(b, z_dim, generator=g)
            out.append(gen(z).squeeze(1).numpy())
            rng_n += b
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-f', action='store_true',
                    help='train on real DS1 F beats via wfdb (default: synthetic mirror)')
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--n-gen', type=int, default=3000)
    ap.add_argument('--out', default='gan_f_beats.npz')
    a = ap.parse_args()
    if a.real_f:
        from src.data import extract_beats_wfdb, DE_CHAZAL_DS1
        X, y, p, rr = extract_beats_wfdb()
        keep = np.isin(p, DE_CHAZAL_DS1) & (y == 3)
        Xf, rrf = X[keep], rr[keep]
        print(f"real DS1-F beats: {len(Xf)}", flush=True)
    else:
        from src.data import synthetic_beats
        X, y, p, rr = synthetic_beats()
        Xf, rrf = X[y == 3], rr[y == 3]
    gen = train_wgan_gp(Xf, steps=a.steps)
    Xs = generate_beats(gen, a.n_gen)
    # RR for synthetic F: resample the real-F RR distribution (timing is not
    # what the GAN models; keep it honest by reusing observed statistics)
    rng = np.random.default_rng(0)
    rs = rrf[rng.integers(0, len(rrf), a.n_gen)]
    np.savez(a.out, X=Xs, rr=rs.astype(np.float32))
    print(f"saved {a.n_gen} GAN F beats -> {a.out}")


if __name__ == '__main__':
    main()
