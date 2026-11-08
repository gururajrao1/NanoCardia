# NanoCardia — Efficient Heartbeat Classifier (MIT-BIH)

Depthwise-separable 1D CNN for N / S / V / F beats. Optimized for efficiency + rare-class recall.

## Why this is better than a vanilla 1D CNN

| Idea | Effect |
|---|---|
| Depthwise-separable conv (MobileNet-style, cf. ArrhythmiNet V1/V2, DeepArrNet, O-WCNN) | ~8x fewer params/FLOPs vs standard conv, 150–300 KB models at ~0.98–0.99 accuracy |
| Global-average-pool + 1 small FC instead of big Flatten+FC | removes 80%+ of params, better inter-patient generalization |
| Enriched RR fusion (pre/post + local-mean ±5 beats + pre/local ratio, 4→16→16 MLP) | timing separates what morphology cannot; fusion rides near-normal rhythm vs premature V |
| QRS-local branch (center 64-crop, narrow kernel → 16-dim) | fusion is decided in local QRS detail; +~350 params, still ~17.5k total |
| Multi-scale stem (parallel k=7 + k=15 convs) | narrow N/S and wide V/F QRS covered in one pass |
| Class-weighted loss (no SMOTE blowup) | fixes ~90% normal imbalance with zero extra data / training cost |
| Random oversampling `--ros` (repeat real S/F beats, no synthetic waveforms) | every training beat stays valid; pairs with focal loss |
| Within-class mixup `--wcmix` (F↔F/S↔S beta blends, hard labels, N excluded) | the only safe waveform augmentation; cross-class N+V blending collapsed real F |
| 188-sample beats @125Hz, z-norm per beat | compatible with Kaggle cut + Kachuee pipeline, cheap |
| Time-shift + noise augment on minority classes only | boosts S/F recall without full-dataset augmentation cost |

## Setup (project-local env)

```powershell
uv venv
uv pip install -r requirements.txt
# or: python -m venv .venv; .venv\Scripts\activate; pip install -r requirements.txt
```

## Run

```powershell
# fast verification on synthetic beats with MIT-BIH-like 90/3/6/1 imbalance:
uv run python -m src.train --synthetic --epochs 15

# weighted-phase loss: ce (default) or focal (gamma=2, class-weighted alpha):
uv run python -m src.train --synthetic --epochs 15 --loss focal

# opt-in train-only SMOTE (S/F upsampled toward V count, test untouched):
uv run python -m src.train --synthetic --epochs 15 --loss focal --smote

# F-mixup (EXPERIMENTAL: raises S but collapses real F 0.39->0.00/0.02
# on patient-split; synthetic N+V mixes look too V-like vs true fusion):
uv run python -m src.train --synthetic --epochs 15 --loss focal --smote --fmix

# recommended research recipe (focal + real-beat oversampling + safe mixup):
uv run python -m src.train --synthetic --epochs 15 --loss focal --ros --wcmix

# offline GAN F beats (per-class unconditional WGAN-GP, DS1-F only):
uv run python -m src.gan --real-f --steps 2000 --n-gen 3000 --out gan_f_beats.npz
uv run python -m src.train --synthetic --epochs 15 --loss focal --ros --ganf gan_f_beats.npz

# real MIT-BIH via wfdb (downloads from PhysioNet, slower):
uv run python -m src.train --real --epochs 5 --patient-split

# real patient-split with the full research recipe:
uv run python -m src.train --real --epochs 10 --patient-split --loss focal --ros --wcmix

# measured on real MIT-BIH DS1->DS2 (100,633 beats, focal + ros + wcmix + GAN F):
# macro 0.514 -> 0.686 (N 0.77, S 0.81, V 0.94, F 0.23) at 17,529 params.
# Best macro to date; S record (0.81); F still below the no-synth best (0.39).

# sequence models (real DS1->DS2, 9-beat windows, focal + window-ros):
# SeqECG: macro 0.596 (F 0.37); +curriculum 0.3: 0.529 (negative, weighting hurt);
# +adversarial record head: 0.680 (F 0.88 new record, S 0.18 collapses) but
# 0.543 on retrain (identical flags+seed) -- run-to-run nondeterminism confirmed.
# SeqECG alone does not beat the single-beat recipe; ensemble of weak members: 0.51.

# seed-shock verdict: keeper 0.686 never reproduced (0.535, 0.543, 0.535).
# Stabilized recipe (val + minibatch + plateau + SWA), 3 seeds, same data/flags:
# 0.496 / 0.496 / 0.510 -> mean 0.501 +- 0.008, F 0.00 every time.
# Tight variance, collapsed level. No-RR ablation: 0.485. ODE-gated F: 0.513
# (F 0.13 -- only nonzero F in the grid). RR worth ~+0.015; ODE beats GAN (0.00).
# Per-record RR norm (stabilized + col_znorm, seed 0, 100,633 beats): no-weights 0.495
# (N 0.91/S 0.08/V 0.99/F 0.00) -> weighted 0.506 (N 0.94/S 0.10/V 0.98/F 0.00), 17,529 params.
# Inside the stabilized band: norm lifts val (0.678) but not test. Val/test disconnect confirmed.
# Self-ONN Q=3 stem (stabilized + --selfonn, seed 0): no-weights 0.508
# (N 0.98/S 0.12/V 0.93/F 0.00) -> weighted 0.520 (N 0.97/S 0.15/V 0.97/F 0.00), 18,265 params.
# Best stabilized test to date (+0.019 over mean) but F still 0.00; Taylor expressivity helps S/V, not F.

# CNN + SeqECG ensemble (train each with --save/--seed, then combine logits):
uv run python -m src.train --real --epochs 10 --patient-split --loss focal --ros --wcmix --ganf gan_f_beats.npz --save cnn.pt --seed 0
uv run python -m src.train --real --epochs 10 --patient-split --seq --loss focal --ros --adv --save seq.pt --seed 0
uv run python -m src.ensemble --cnn cnn.pt --seq seq.pt --real --patient-split --w-cnn 0.5 --w-seq 0.5

# maintained checks (no extra framework):
uv run python -m tests.test_upgrades
```

Outputs per-class recall, macro-recall, param count, inference time/beat.
